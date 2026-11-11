#!/usr/bin/env python3
"""
印刷调色 Web 服务（基于 http.server + 线程池）
模型：CurvePredictor（v3，checkpoints/curve_pred_best.pth）
输出：RGB 预览 + CMYK 印刷稿（PSOcoated_v3.icc）
"""
import argparse, json, mimetypes, os, shutil, threading, uuid, zipfile, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image

from processor import (process_file, load_model,
                       apply_cmyk_curves, cmyk_to_srgb, curves_are_identity,
                       normalize_curves, save_cmyk_image, save_cmyk_pdf,
                       apply_cmyk_curves_to_pdf,
                       cmyk_container_name, is_pdf, pdf_page_count,
                       load_first_page_rgb, pil_to_bytes)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}
DEFAULT_MODEL = "checkpoints/curve_pred_best.pth"
PROOF_SIZE = 1400          # 软打样底图边长，用于曲线面板实时预览
PREVIEW_SIZE = 1600
BATCH_DIRS = ("input", "input_preview", "output", "pages", "stages", "adjusted", "proof",
              "preview", "target", "target_preview")
DEFAULT_GAINS = {"curve": 1.2, "skin": 1.2, "residual": 1.0}


def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")


def original_filename(name):
    """保留上传时的完整文件名与原始后缀（含大小写），仅去掉路径与非法字符。"""
    name = Path(unquote(name)).name.strip().replace("\x00", "")
    name = name.replace("/", "").replace("\\", "")
    if not name or name in {".", ".."}:
        raise ValueError("非法文件名")
    suffix = Path(name).suffix
    if suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("仅支持 JPG、PNG 和 TIFF 图片")
    return name[:180]


def stored_name_with_suffix(image_id, filename):
    return f"{image_id}{Path(filename).suffix}"


def download_filename(original_name, output_path):
    """下载名与输出文件后缀一致：JPG/TIF 保持原名，PNG 改为同名 .tif。"""
    original_name = original_name or Path(output_path).name
    out_suffix = Path(output_path).suffix
    if Path(original_name).suffix.lower() == out_suffix.lower():
        return original_name
    return f"{Path(original_name).stem}{out_suffix}"


def unique_zip_name(filename, image_id, used):
    name = filename or "image"
    if name not in used:
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    return f"{stem}_{image_id}{suffix}"


class AppState:
    def __init__(self, model_path, data_dir, workers=2, max_upload_mb=512,
                 inference_size=512, pdf_dpi=300):
        self.model = load_model(model_path)
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.max_upload_mb = max_upload_mb
        self.max_upload_bytes = max_upload_mb * 1024 * 1024
        self.inference_size = inference_size              # 曲线忽略，仅兼容
        self.pdf_dpi = pdf_dpi
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.infer_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="colour")

    def batch_dir(self, batch_id):
        if not batch_id.replace("-", "").isalnum():
            raise ValueError("非法批次 ID")
        return self.data_dir / batch_id

    def manifest_path(self, batch_id): return self.batch_dir(batch_id) / "manifest.json"

    def read_batch(self, batch_id):
        with self.lock:
            p = self.manifest_path(batch_id)
            if not p.exists(): raise FileNotFoundError(batch_id)
            return json.loads(p.read_text("utf-8"))

    def write_batch(self, manifest):
        with self.lock:
            p = self.manifest_path(manifest["id"])
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
            tmp.replace(p)

    def create_batch(self, name):
        batch_id = uuid.uuid4().hex[:12]
        root = self.batch_dir(batch_id)
        for child in BATCH_DIRS:
            (root / child).mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": batch_id,
            "name": name.strip()[:80] or f"调色批次 {datetime.now():%m-%d %H:%M}",
            "created_at": utc_now(), "updated_at": utc_now(),
            "model": self.model_path.name,
            "target_profile": "CMYK",
            "gains": dict(DEFAULT_GAINS),
            "curves": normalize_curves(None),
            "curves_status": "idle",
            "images": [],
        }
        self.write_batch(manifest)
        return manifest

    def list_batches(self):
        rows = []
        for p in self.data_dir.glob("*/manifest.json"):
            try:
                b = json.loads(p.read_text("utf-8"))
                rows.append({k: b[k] for k in ("id", "name", "created_at", "updated_at")}
                            | {"counts": self.counts(b), "target_profile": b.get("target_profile")})
            except (OSError, ValueError, KeyError):
                continue
        return sorted(rows, key=lambda x: x["created_at"], reverse=True)

    @staticmethod
    def counts(batch):
        imgs = batch["images"]
        return {
            "total": len(imgs),
            "completed": sum(x["process_status"] == "completed" for x in imgs),
            "processing": sum(x["process_status"] in {"queued", "processing"} for x in imgs),
            "failed": sum(x["process_status"] == "failed" for x in imgs),
            "approved": sum(x["review_status"] == "approved" for x in imgs),
            "rejected": sum(x["review_status"] == "rejected" for x in imgs),
            "pending": sum(x["review_status"] == "pending" for x in imgs),
        }

    def add_upload_file(self, batch_id, filename, uploaded_path, size_bytes):
        filename = original_filename(filename)
        image_id = uuid.uuid4().hex[:12]
        stored_name = stored_name_with_suffix(image_id, filename)
        final_path = self.batch_dir(batch_id) / "input" / stored_name
        pdf = is_pdf(filename) or is_pdf(uploaded_path)
        try:
            if pdf:
                page_count = pdf_page_count(uploaded_path)
                first = load_first_page_rgb(uploaded_path)
                height, width = first.shape[:2]
            else:
                page_count = 1
                with Image.open(uploaded_path) as img:
                    img.load()
                    width, height = img.size
        except Exception as e:
            err = uploaded_path.with_suffix(".error")
            uploaded_path.rename(err)
            kind = "PDF" if pdf else "图片"
            raise ValueError(f"上传内容不是有效{kind}: {e} ({err})")
        uploaded_path.replace(final_path)

        with self.lock:
            batch = self.read_batch(batch_id)
            item = {
                "id": image_id, "filename": filename,
                "input_file": stored_name, "output_file": None, "preview_file": None,
                "target_file": None, "target_preview_file": None, "target_filename": None,
                "width": width, "height": height, "size_bytes": size_bytes,
                "process_status": "queued", "review_status": "pending",
                "note": "", "error": None, "created_at": utc_now(),
                "processed_at": None, "metrics": None,
                "adjusted_file": None, "proof_file": None,
                "is_pdf": pdf, "page_count": page_count, "pages": [],
            }
            batch["images"].append(item)
            batch["updated_at"] = utc_now()
            self.write_batch(batch)
        self.executor.submit(self.process_image, batch_id, image_id)
        return item

    def update_item(self, batch_id, image_id, **changes):
        with self.lock:
            batch = self.read_batch(batch_id)
            item = next((x for x in batch["images"] if x["id"] == image_id), None)
            if not item: raise FileNotFoundError(image_id)
            item.update(changes)
            batch["updated_at"] = utc_now()
            self.write_batch(batch)
            return item.copy()

    def process_image(self, batch_id, image_id):
        item = self.update_item(batch_id, image_id, process_status="processing", error=None)
        root = self.batch_dir(batch_id)
        try:
            input_path = root / "input" / item["input_file"]
            target_path = None
            if item.get("target_file"):
                target_path = root / "target" / item["target_file"]

            output_name = stored_name_with_suffix(image_id, item["filename"])
            gains = (self.read_batch(batch_id).get("gains") or dict(DEFAULT_GAINS))
            with self.infer_lock:
                result = process_file(
                    image_path=str(input_path),
                    model=self.model,
                    output_dir=str(root / "output"),
                    output_filename=output_name,
                    target_path=str(target_path) if target_path else None,
                    return_metrics=True,
                    proof_dir=str(root / "proof"),
                    input_preview_dir=str(root / "input_preview"),
                    page_dir=str(root / "pages"),
                    stage_dir=str(root / "stages"),
                    page_stem=image_id,
                    proof_size=PROOF_SIZE,
                    input_preview_size=PREVIEW_SIZE,
                    pdf_dpi=self.pdf_dpi,
                    gains=gains,
                )

            out_path = Path(result["output_path"])
            first_page = result["pages"][0] if result["pages"] else {}
            self.update_item(
                batch_id, image_id,
                output_file=out_path.name,
                proof_file=first_page.get("proof_file"),
                pages=result["pages"],
                page_count=result["page_count"],
                dpi=result.get("dpi"),
                # 以实际渲染尺寸为准（PDF 的 DPI 由服务端决定）
                width=first_page.get("width") or item.get("width"),
                height=first_page.get("height") or item.get("height"),
                process_status="completed",
                processed_at=utc_now(),
                metrics=result["metrics"],
                review_status="pending" if target_path else "approved",
                cmyk_file=out_path.name,
            )
            self.render_item_outputs(batch_id, image_id)
        except Exception as exc:
            traceback.print_exc()
            self.update_item(batch_id, image_id, process_status="failed", error=str(exc))

    # ---- CMYK 手工曲线：交付图与软打样预览 ----
    @staticmethod
    def item_pages(item):
        """兼容早期单页数据：始终返回页列表。"""
        pages = item.get("pages")
        if pages:
            return pages
        if not item.get("output_file"):
            return []
        return [{"index": 0, "width": item.get("width"), "height": item.get("height"),
                 "proof_file": item.get("proof_file")}]

    def render_item_outputs(self, batch_id, image_id, curves=None):
        """按批次曲线，从基准 CMYK 输出生成交付文件与逐页 sRGB 软打样预览。"""
        batch = self.read_batch(batch_id)
        if curves is None:
            curves = batch.get("curves")
        item = next((x for x in batch["images"] if x["id"] == image_id), None)
        if not item or not item.get("output_file"):
            return
        root = self.batch_dir(batch_id)
        base_path = root / "output" / item["output_file"]
        if not base_path.exists():
            return
        identity = curves_are_identity(curves)
        pages = self.item_pages(item)

        # 1) 交付文件：逐页套曲线，PDF 重组为多页 CMYK PDF
        adjusted_dir = root / "adjusted"
        adjusted_dir.mkdir(parents=True, exist_ok=True)
        adjusted_name = None
        if identity:
            old = item.get("adjusted_file")
            if old:
                (adjusted_dir / old).unlink(missing_ok=True)
        else:
            adjusted_name = item["output_file"]
            target = adjusted_dir / adjusted_name
            base_pages = [p.get("base_page_file") for p in pages if p.get("base_page_file")]
            pdfish = is_pdf(base_path) or is_pdf(item.get("filename", ""))
            layered_ok = False
            if pdfish:
                try:
                    apply_cmyk_curves_to_pdf(str(base_path), str(target), curves)
                    layered_ok = True
                except Exception:
                    traceback.print_exc()
                    print("   分层套曲线失败，改用整页 TIFF 重组")
            if layered_ok:
                pass
            elif base_pages:                      # 旧批次或超大 PDF 回退：逐页 TIFF 重组
                tmp_dir = adjusted_dir / f".{image_id}_pages"
                shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                try:
                    page_files = []
                    for index, name in enumerate(base_pages):
                        with Image.open(root / "pages" / name) as img:
                            img.load()
                            page_path = tmp_dir / f"p{index}.tif"
                            save_cmyk_image(apply_cmyk_curves(img, curves), page_path, lossless=True)
                        page_files.append(page_path)
                    save_cmyk_pdf(
                        page_files, target,
                        dpi=item.get("dpi") or 300,
                        page_sizes_pt=[p.get("size_pt") for p in pages],
                    )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                with Image.open(base_path) as img:
                    img.load()
                    save_cmyk_image(apply_cmyk_curves(img, curves), target)

        # 2) 网页预览：软打样底图来自分层成功后重新栅格化的交付 PDF
        preview_dir = root / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        updated_pages = []
        for page in pages:
            page_meta = dict(page)
            proof_name = page.get("proof_file")
            proof_path = root / "proof" / proof_name if proof_name else None
            if not (proof_path and proof_path.is_file()):
                updated_pages.append(page_meta)
                continue
            preview_name = f"{image_id}_p{page.get('index', 0)}.jpg"
            with Image.open(proof_path) as img:
                img.load()
                proofed = cmyk_to_srgb(apply_cmyk_curves(img, curves))
            proofed.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
            proofed.save(preview_dir / preview_name, quality=91, optimize=True)
            page_meta["preview_file"] = preview_name
            updated_pages.append(page_meta)

        first_preview = next((p.get("preview_file") for p in updated_pages if p.get("preview_file")), None)
        return self.update_item(batch_id, image_id,
                                adjusted_file=adjusted_name,
                                pages=updated_pages,
                                preview_file=first_preview)


    def set_curves(self, batch_id, curves):
        """保存批次曲线，并在后台对整批已完成图片重新出图。"""
        normalized = normalize_curves(curves)
        with self.lock:
            batch = self.read_batch(batch_id)
            batch["curves"] = normalized
            batch["curves_status"] = "applying"
            batch["updated_at"] = utc_now()
            self.write_batch(batch)
        self.executor.submit(self._apply_curves_job, batch_id, normalized)
        return {"ok": True, "curves": normalized, "curves_status": "applying"}

    def _apply_curves_job(self, batch_id, curves):
        try:
            batch = self.read_batch(batch_id)
            for item in batch["images"]:
                if item.get("process_status") != "completed":
                    continue
                try:
                    self.render_item_outputs(batch_id, item["id"], curves)
                except Exception:
                    traceback.print_exc()
        finally:
            with self.lock:
                try:
                    batch = self.read_batch(batch_id)
                except FileNotFoundError:
                    batch = None
                if batch is not None:
                    batch["curves_status"] = "idle"
                    batch["updated_at"] = utc_now()
                    self.write_batch(batch)

    def curve_preview(self, batch_id, image_id, curves, size=900, page=0):
        """把曲线套在指定页的软打样底图上，返回可直接显示的 JPEG。"""
        batch = self.read_batch(batch_id)
        candidates = [x for x in batch["images"] if x.get("output_file")]
        if not candidates:
            raise FileNotFoundError("批次中还没有已完成的输出")
        item = next((x for x in candidates if x["id"] == image_id), candidates[0])
        pages = self.item_pages(item)
        try:
            page_index = max(0, min(len(pages) - 1, int(page)))
        except (TypeError, ValueError):
            page_index = 0
        proof_name = pages[page_index].get("proof_file") if pages else None
        root = self.batch_dir(batch_id)
        proof_path = root / "proof" / proof_name if proof_name else None
        if not (proof_path and proof_path.is_file()):
            raise FileNotFoundError("软打样底图不存在")
        size = max(200, min(1600, int(size)))
        with Image.open(proof_path) as img:
            img.load()
            proofed = cmyk_to_srgb(apply_cmyk_curves(img, curves))
        proofed.thumbnail((size, size), Image.Resampling.LANCZOS)
        raw, mime = pil_to_bytes(proofed, fmt="JPEG", quality=88)
        return raw, mime, item["id"]

    def delivery_path(self, batch_id, item):
        """下载用的最终 CMYK 文件：优先曲线调整后的版本。"""
        root = self.batch_dir(batch_id)
        if item.get("adjusted_file"):
            adjusted = root / "adjusted" / item["adjusted_file"]
            if adjusted.exists():
                return adjusted
        return root / "output" / (item.get("output_file") or "")

    def review(self, batch_id, image_id, status, note):
        if status not in {"pending", "approved", "rejected"}:
            raise ValueError("非法审核状态")
        return self.update_item(batch_id, image_id, review_status=status, note=note[:500])

    def delete_batch(self, batch_id):
        root = self.batch_dir(batch_id)
        with self.lock:
            if not self.manifest_path(batch_id).exists(): raise FileNotFoundError(batch_id)
            shutil.rmtree(root)
        return {"ok": True, "id": batch_id}

    def add_target_file(self, batch_id, image_id, filename, uploaded_path):
        filename = original_filename(filename)
        batch = self.read_batch(batch_id)
        item = next((x for x in batch["images"] if x["id"] == image_id), None)
        if not item: raise FileNotFoundError(image_id)
        root = self.batch_dir(batch_id)
        target_dir, preview_dir = root / "target", root / "target_preview"
        target_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        stored_name, preview_name = stored_name_with_suffix(image_id, filename), image_id + ".jpg"
        path = target_dir / stored_name

        try:
            if (item.get("page_count") or 1) > 1:
                raise ValueError("多页 PDF 暂不支持上传目标图做指标比对")
            target_pil = Image.fromarray(load_first_page_rgb(uploaded_path))
            if target_pil.size != (item["width"], item["height"]):
                raise ValueError(
                    f"目标图尺寸必须与原图一致：需要 {item['width']}×{item['height']}，"
                    f"实际为 {target_pil.width}×{target_pil.height}")
            target_pil.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
            target_pil.save(preview_dir / preview_name, quality=91, optimize=True)
        except Exception as e:
            (preview_dir / preview_name).unlink(missing_ok=True)
            raise

        old = item.get("target_file")
        if old and old != stored_name:
            (target_dir / old).unlink(missing_ok=True)
        uploaded_path.replace(path)
        self.update_item(batch_id, image_id,
                         target_file=stored_name, target_preview_file=preview_name,
                         target_filename=filename)
        self.executor.submit(self.process_image, batch_id, image_id)
        return self.read_batch(batch_id)["images"][-1]

    def make_zip(self, batch_id, status):
        batch = self.read_batch(batch_id)
        if status not in {"all", "approved", "rejected", "pending"}:
            raise ValueError("非法筛选条件")
        path = self.batch_dir(batch_id) / f"{status}_outputs.zip"
        used = set()
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
            for item in batch["images"]:
                if not item.get("output_file") or (status != "all" and item["review_status"] != status):
                    continue
                src = self.delivery_path(batch_id, item)
                if not src.exists():
                    continue
                arcname = unique_zip_name(
                    download_filename(item.get("filename"), src),
                    item["id"],
                    used,
                )
                used.add(arcname)
                zf.write(src, arcname=arcname)
            zf.writestr("review_manifest.json", json.dumps(batch, ensure_ascii=False, indent=2))
        return path


# ---- HTTP Handler ----
class Handler(BaseHTTPRequestHandler):
    server_version = "ColorReview/1.0"

    @property
    def app(self): return self.server.app

    def log_message(self, fmt, *args): print(f"[{self.log_date_time_string()}] {fmt % args}")

    def json_response(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def error_response(self, msg, status=400): self.json_response({"error": msg}, status)

    def bytes_response(self, raw, mime):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024: raise ValueError("请求过大")
        return json.loads(self.rfile.read(length) or b"{}")

    def receive_upload(self, path, length):
        remaining = length
        with path.open("wb") as out:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk: raise ConnectionError("上传连接提前中断")
                out.write(chunk)
                remaining -= len(chunk)

    def send_file(self, path, download_name=None, inline=False):
        if not path.exists() or not path.is_file():
            self.send_error(404); return
        name_for_type = download_name or path.name
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(name_for_type)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            ascii_name = download_name.encode("ascii", "replace").decode("ascii").replace('"', "")
            kind = "inline" if inline else "attachment"
            self.send_header(
                "Content-Disposition",
                f'{kind}; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(download_name)}',
            )
        self.end_headers()
        with path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            parts = [x for x in parsed.path.split("/") if x]

            if parsed.path == "/api/config":
                self.json_response({"max_upload_mb": self.app.max_upload_mb,
                                    "max_upload_bytes": self.app.max_upload_bytes})
            elif parsed.path == "/api/batches":
                self.json_response(self.app.list_batches())
            elif len(parts) == 3 and parts[:2] == ["api", "batches"]:
                batch = self.app.read_batch(parts[2])
                batch["counts"] = self.app.counts(batch)
                self.json_response(batch)
            elif len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "download":
                status = parse_qs(parsed.query).get("status", ["approved"])[0]
                self.send_file(self.app.make_zip(parts[2], status), f"{parts[2]}_{status}.zip")
            elif len(parts) == 5 and parts[:2] == ["api", "batches"] and parts[4] == "download":
                batch_id, image_id = parts[2], parts[3]
                fmt = parse_qs(parsed.query).get("format", ["original"])[0]
                batch = self.app.read_batch(batch_id)
                item = next((x for x in batch["images"] if x["id"] == image_id), None)
                if not item:
                    raise FileNotFoundError("图片不存在")

                if not item.get("output_file"):
                    raise FileNotFoundError("输出文件尚未生成，请确认处理已完成")
                if fmt == "base":
                    output_path = self.app.batch_dir(batch_id) / "output" / item["output_file"]
                else:
                    output_path = self.app.delivery_path(batch_id, item)
                self.send_file(output_path, download_filename(item.get("filename"), output_path))
            elif len(parts) == 4 and parts[0] == "media":
                batch = self.app.read_batch(parts[1])
                item = next((x for x in batch["images"] if x["id"] == parts[2]), None)
                if not item:
                    raise FileNotFoundError("图片不存在")
                query = parse_qs(parsed.query)
                inline = query.get("inline", ["0"])[0] not in {"0", "", "false"}
                page_index = query.get("page", ["0"])[0]
                root = self.app.batch_dir(parts[1])

                if parts[3] in {"output", "cmyk"}:
                    if not item.get("output_file"):
                        raise FileNotFoundError(parts[3])
                    file_path = self.app.delivery_path(parts[1], item)
                    self.send_file(file_path, download_filename(item.get("filename"), file_path),
                                   inline=inline)
                    return
                if parts[3] in {"preview", "input-preview"}:
                    pages = self.app.item_pages(item)
                    try:
                        idx = max(0, min(len(pages) - 1, int(page_index)))
                    except (TypeError, ValueError):
                        idx = 0
                    key = "preview_file" if parts[3] == "preview" else "input_preview_file"
                    folder = "preview" if parts[3] == "preview" else "input_preview"
                    name = pages[idx].get(key) if pages else None
                    if not name:
                        name = item.get("preview_file") if parts[3] == "preview" else None
                    if not name:
                        raise FileNotFoundError(parts[3])
                    self.send_file(root / folder / name)
                    return
                mapping = {
                    "input": ("input", "input_file"),
                    "target": ("target", "target_file"),
                    "target-preview": ("target_preview", "target_preview_file"),
                }
                if parts[3] not in mapping or not item.get(mapping[parts[3]][1]):
                    raise FileNotFoundError(parts[3])
                folder, key = mapping[parts[3]]
                file_path = root / folder / item[key]
                if parts[3] == "input" and inline:
                    self.send_file(file_path, item.get("filename"), inline=True)
                else:
                    self.send_file(file_path)
            elif parsed.path.startswith("/api/"):
                self.send_error(404)
            else:
                static_root = Path(__file__).parent / "web"
                if not static_root.exists(): static_root = Path(__file__).parent / "static"
                request_path = parsed.path
                file_name = request_path[8:] if request_path.startswith("/static/") else request_path.lstrip("/") or "index.html"
                if request_path in ("/", "/review"): file_name = "index.html"
                file_path = (static_root / file_name).resolve()
                file_path.relative_to(static_root)   # 防目录穿越
                if not file_path.exists(): self.send_error(404); return
                self.send_file(file_path)
        except FileNotFoundError: self.error_response("未找到资源", 404)
        except (ValueError, KeyError, StopIteration) as e: self.error_response(str(e), 400)
        except Exception as e: self.error_response(f"服务器错误：{e}", 500)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            parts = [x for x in parsed.path.split("/") if x]
            if parsed.path == "/api/batches":
                self.json_response(self.app.create_batch(self.read_json().get("name", "")), 201); return
            if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "images":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > self.app.max_upload_bytes: raise ValueError("单张图片超限")
                filename = parse_qs(parsed.query).get("filename", [""])[0]
                if not filename: raise ValueError("缺少 filename 参数")
                upload_dir = self.app.batch_dir(parts[2]) / "input"
                upload_dir.mkdir(parents=True, exist_ok=True)
                tmp = upload_dir / f".{uuid.uuid4().hex}.upload"
                try:
                    self.receive_upload(tmp, length)
                    item = self.app.add_upload_file(parts[2], filename, tmp, length)
                finally:
                    tmp.unlink(missing_ok=True)
                self.json_response(item, 202); return
            if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "curves":
                data = self.read_json()
                self.json_response(self.app.set_curves(parts[2], data.get("curves")), 202); return
            if (len(parts) == 5 and parts[:2] == ["api", "batches"]
                    and parts[3] == "curves" and parts[4] == "preview"):
                data = self.read_json()
                raw, mime, _ = self.app.curve_preview(
                    parts[2], data.get("image"), data.get("curves"),
                    data.get("size", 900), data.get("page", 0))
                self.bytes_response(raw, mime); return
            if len(parts) == 5 and parts[:2] == ["api", "batches"] and parts[4] == "target":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > self.app.max_upload_bytes: raise ValueError("目标图超限")
                filename = parse_qs(parsed.query).get("filename", [""])[0]
                if not filename: raise ValueError("缺少 filename 参数")
                upload_dir = self.app.batch_dir(parts[2]) / "target"
                upload_dir.mkdir(parents=True, exist_ok=True)
                tmp = upload_dir / f".{parts[3]}.{uuid.uuid4().hex}.upload"
                try:
                    self.receive_upload(tmp, length)
                    item = self.app.add_target_file(parts[2], parts[3], filename, tmp)
                finally:
                    tmp.unlink(missing_ok=True)
                self.json_response(item, 201); return
            self.send_error(404)
        except FileNotFoundError: self.error_response("批次不存在", 404)
        except (ValueError, json.JSONDecodeError) as e: self.error_response(str(e), 400)
        except Exception as e: self.error_response(f"服务器错误：{e}", 500)

    def do_PATCH(self):
        try:
            parts = [x for x in urlparse(self.path).path.split("/") if x]
            if len(parts) != 5 or parts[:2] != ["api", "batches"] or parts[4] != "review":
                self.send_error(404); return
            data = self.read_json()
            item = self.app.review(parts[2], parts[3], data.get("status", "pending"), data.get("note", ""))
            self.json_response(item)
        except FileNotFoundError: self.error_response("图片不存在", 404)
        except (ValueError, json.JSONDecodeError) as e: self.error_response(str(e), 400)

    def do_DELETE(self):
        try:
            parts = [x for x in urlparse(self.path).path.split("/") if x]
            if len(parts) != 3 or parts[:2] != ["api", "batches"]: self.send_error(404); return
            self.json_response(self.app.delete_batch(parts[2]))
        except FileNotFoundError: self.error_response("批次不存在", 404)
        except ValueError as e: self.error_response(str(e), 400)


def main():
    ap = argparse.ArgumentParser(description="印刷调色 Web 服务（CurvePredictor v3）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="CurvePredictor 权重路径")
    ap.add_argument("--data", default="web_data", help="数据存储目录")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-upload-mb", type=int, default=512)
    ap.add_argument("--inference-size", type=int, default=512, help="(曲线模型忽略，仅兼容接口)")
    ap.add_argument("--pdf-dpi", type=int, default=300,
                    help="输入 PDF 没有嵌入图时的默认栅格化 DPI；有嵌入图时跟输入精度走")
    args = ap.parse_args()

    app = AppState(Path(args.model), Path(args.data),
                    workers=args.workers, max_upload_mb=args.max_upload_mb,
                    inference_size=args.inference_size, pdf_dpi=args.pdf_dpi)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app = app
    print(f"服务运行在 http://{args.host}:{args.port}")
    print(f"模型: {Path(args.model).resolve()}")
    print(f"数据目录: {Path(args.data).resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        app.executor.shutdown(wait=True)
        server.server_close()


if __name__ == "__main__":
    main()