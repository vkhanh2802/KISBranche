# AIC 2026 KIS Retrieval

Hệ thống tìm kiếm keyframe video bằng ba luồng retrieval và một VLM reranker:

- SigLIP visual search trên Milvus.
- BM25 keyword search trên caption.
- SentenceTransformer semantic search trên caption.
- Object metadata boost nếu có dữ liệu object detection.
- Qwen2.5-VL rerank top 20 kết quả.
- FastAPI web app để nhập raw query và xem ảnh top 20.

## Yêu cầu quan trọng

Clone GitHub **chưa đủ để chạy hệ thống**. Các thư mục dữ liệu và artifact lớn
được bỏ qua trong `.gitignore`, vì vậy người duy trì project phải cung cấp chúng
qua Google Drive, Hugging Face Dataset, object storage hoặc một kênh riêng:

```text
data/
mapping/
clip-features-32/
objects/                 # tùy chọn
caption_generator/       # có thể tự build
caption_mapping/         # có thể tự build
index/                   # có thể tự build
```

Project hiện tại không có bước tách raw video thành keyframe, tạo file mapping
hoặc sinh `clip-features-32/*.npy`. Do đó `data/`, `mapping/` và
`clip-features-32/` là ba đầu vào bắt buộc phải nhận từ nhà cung cấp dataset.

Milvus cũng là thành phần bắt buộc. Cách tái lập dễ nhất là dùng Docker Compose
đã được kèm trong repository. Nếu đã có Milvus v3 tương thích đang nghe tại
`http://localhost:19530`, không cần khởi động thêm Compose.

## Yêu cầu hệ thống

- Windows 10/11 hoặc Linux 64-bit.
- Python 3.11. Khuyến nghị Miniconda/Anaconda.
- Git.
- Docker Desktop có Docker Compose để chạy Milvus.
- NVIDIA GPU và CUDA-compatible driver nếu muốn dùng Qwen VLM.
- Khoảng 6 GB VRAM cho chế độ VLM 4-bit hiện tại. Có thể tắt VLM nếu không có GPU.
- Dung lượng đĩa còn trống cho dataset, Milvus và Hugging Face model cache.
- Gemini API key để chuyển raw query thành structured query.
- Internet trong lần đầu để tải model theo revision trong `model_manifest.json`.

## 1. Clone repository

```powershell
git clone <REPOSITORY_URL>
cd KISBranche
```

Thay `<REPOSITORY_URL>` bằng URL GitHub của repository này.

## 2. Tạo môi trường Python

Trên PowerShell/Anaconda Prompt:

```powershell
conda create -n focusguard_ai python=3.11 -y
conda activate focusguard_ai
```

Nếu dùng VLM trên NVIDIA GPU, cài bản PyTorch CUDA phù hợp với máy từ
[pytorch.org](https://pytorch.org/get-started/locally/) trước. Ví dụ với CUDA
12.8 và các version hiện tại của project:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

Sau đó cài các dependency còn lại:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

Nếu chỉ chạy retrieval không VLM, bản PyTorch CPU vẫn dùng được, nhưng cần tắt
VLM khi khởi động app ở bước 8.

## 3. Đặt dữ liệu đúng cấu trúc

Keyframe có thể theo layout grouped (khuyến nghị):

```text
data/
  L21/
    L21_V001/
      001.jpg
      002.jpg
      ...
  L27/
    L27_V013/
      001.jpg
      ...
```

Layout phẳng `data/L21_V001/001.jpg` cũng được hỗ trợ, nhưng không được đặt cùng
một video trong cả hai layout.

Mỗi video cần mapping CSV và visual embedding cùng tên:

```text
mapping/
  L21_V001.csv
  L27_V013.csv

clip-features-32/
  L21_V001.npy
  L27_V013.npy
```

Mapping CSV phải có các cột:

```csv
n,pts_time,fps,frame_idx
1,0.0,25.0,0
2,0.04,25.0,1
```

Mỗi `.npy` phải có shape `(số_keyframe, 768)`, cùng thứ tự với mapping CSV và
được sinh bởi model SigLIP/revision ghi trong `model_manifest.json`.

Object metadata là tùy chọn:

```text
objects/
  L21_V001/
    001.json
    002.json
```

Nếu nhà cung cấp đã gửi cả caption artifact build sẵn, đặt chúng như sau và bỏ
qua bước 5:

```text
caption_generator/<group>/<video_id>.json
caption_mapping/<group>/<video_id>.json
index/<group>/<video_id>.index
index/<group>/<video_id>.meta.json
```

Không commit các thư mục trên vào Git nếu không có quyền phân phối dataset.

## 4. Khởi động Milvus bằng Docker Compose

Docker Compose được track tại `docker-compose.yml`; người clone repo không cần
tự viết một compose khác.

```powershell
docker compose up -d
docker compose ps
```

Đợi đến khi `milvus-etcd`, `milvus-minio` và `milvus-standalone` đều healthy.
Health endpoint của Milvus là `http://localhost:9091/healthz`; SDK kết nối qua
port `19530`.

Dữ liệu container được lưu tại `volumes/` và không được commit. Nếu máy đã có
stack Milvus chiếm các port hoặc tên container này, hãy dùng stack đang có thay
vì khởi động stack thứ hai.

## 5. Tạo caption artifact

Bỏ qua mục này nếu đã nhận đủ bộ bốn artifact caption/index build sẵn cho mỗi
video.

Tạo caption và VQA metadata từ toàn bộ keyframe:

```powershell
python caption_generator.py
```

Lệnh này tải BLIP caption và BLIP VQA theo revision đã pin, có thể mất nhiều giờ
với dataset lớn. Script checkpoint theo từng video/batch và bỏ qua frame đã có
caption hợp lệ.

OCR bằng Gemini là tùy chọn. Chỉ chạy nếu visible text là tín hiệu quan trọng:

```powershell
python gemini_ocr.py
```

OCR gửi ảnh lên Gemini, có chi phí/rate limit và có khoảng nghỉ giữa các request.
Nếu chạy OCR, phải chạy nó trước khi build caption index.

Build FAISS caption index, mapping và integrity manifest:

```powershell
python build_caption_index.py
```

Build lại một số video cụ thể:

```powershell
python build_caption_index.py --video-id L27_V013 --video-id L27_V014
```

Nếu đang chuyển artifact từ layout legacy, xem dry-run trước:

```powershell
python migrate_caption_artifacts.py
python migrate_caption_artifacts.py --apply
```

## 6. Build visual collection trong Milvus

Lần đầu trên một Milvus rỗng:

```powershell
python build_milvus.py build
```

Lệnh này validate toàn bộ `.npy` và mapping trước, sau đó publish collection qua
alias `clip_keyframes`. Khi thay đổi visual feature và cần thay collection hiện
tại một cách an toàn:

```powershell
python build_milvus.py rebuild
```

Không cần build lại Milvus mỗi lần mở app. Collection tồn tại trong `volumes/`
cho đến khi thư mục đó bị xóa.

## 7. Cấu hình Gemini API key

Khuyến nghị chỉ set key trong terminal đang chạy app.

PowerShell:

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Linux/macOS:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Kiểm tra mà không in key ra màn hình:

```powershell
if ($env:GEMINI_API_KEY) { "GEMINI_API_KEY is set" } else { "GEMINI_API_KEY is missing" }
```

Không ghi API key vào source code, notebook, commit hoặc issue GitHub.

## 8. Chạy web app

Đảm bảo Docker/Milvus đang chạy, môi trường `focusguard_ai` đang active và API
key đã được set trong cùng terminal:

```powershell
python web_app.py
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000), nhập raw query và nhấn
`Tìm top 20`. Mỗi card hiển thị ảnh, rank, `video_id`, số image/keyframe,
`frame_id`, score và caption.

Lần tìm đầu tiên sẽ tải/nạp SigLIP, SentenceTransformer, FAISS caption index,
BM25 và Qwen VLM. Các lần sau tái sử dụng model trong RAM. Chỉ chạy một Uvicorn
worker, nếu không mỗi worker sẽ nạp một bản model và có thể làm hết VRAM.

Máy không có NVIDIA GPU hoặc không đủ VRAM:

```powershell
$env:KIS_DISABLE_VLM = "1"
python web_app.py
```

Chế độ này vẫn chạy visual + BM25 + semantic + object fusion, chỉ bỏ qua Qwen
VLM reranking.

Dừng app bằng `Ctrl+C`. Có thể dừng Milvus mà không xóa dữ liệu:

```powershell
docker compose stop
```

## 9. Quy trình chạy hằng ngày

Sau khi đã build artifact và Milvus một lần:

```powershell
conda activate focusguard_ai
docker compose up -d
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
python web_app.py
```

Không chạy lại `caption_generator.py`, `build_caption_index.py` hay
`build_milvus.py` nếu dữ liệu nguồn không thay đổi.

## 10. Chạy bằng notebook

Web app là cách sử dụng đề xuất. Notebook vẫn hữu ích khi debug từng bước:

```powershell
python structure_query.py "<raw query>" --query-id q_001
```

Sau đó mở `runbranch.ipynb` bằng VS Code hoặc Jupyter. Trong notebook, chạy các
cell init model một lần. Sau đó có thể ghi đè
`structured_query.json` và chạy lại từ cell `LOAD/RELOAD QUERY` đến cuối.

## 11. Kiểm thử

Sau khi đã đặt dataset/artifact mẫu:

```powershell
python -m unittest discover
```

Một số test dữ liệu/FAISS sẽ skip hoặc không thể pass nếu dependency tùy chọn
hay sample artifact chưa được cung cấp.

## Troubleshooting

### `Missing GEMINI_API_KEY environment variable`

Set lại `$env:GEMINI_API_KEY` trong đúng terminal sẽ chạy `python web_app.py`.
Biến set ở terminal khác không tự động truyền sang process hiện tại.

### Không kết nối được `localhost:19530`

```powershell
docker compose ps
docker compose logs standalone
```

Đảm bảo container healthy và port `19530` không bị service khác chiếm.

### `clip_keyframes` không tồn tại hoặc sai model metadata

Đảm bảo `clip-features-32/` và `mapping/` đúng cặp, sau đó chạy trên Milvus rỗng:

```powershell
python build_milvus.py build
```

Nếu collection cũ đã tồn tại và visual feature đã thay đổi, dùng `rebuild`.

### Không tìm thấy caption/index

Mỗi video cần đủ caption JSON, caption mapping JSON, FAISS index và manifest.
Nếu chỉ có ảnh + mapping, chạy:

```powershell
python caption_generator.py
python build_caption_index.py
```

### `VLM requires CUDA by default`

Cài đúng PyTorch CUDA/driver, hoặc tắt VLM:

```powershell
$env:KIS_DISABLE_VLM = "1"
python web_app.py
```

### CUDA out of memory

Đóng các process đang dùng GPU. Nếu vẫn thiếu VRAM, tắt VLM như trên. Qwen VLM
hiện được nạp 4-bit để giảm VRAM, nhưng vẫn cần bộ nhớ cho image tokens và KV
cache.

## Artifact và model reproducibility

Model ID, immutable revision, vector dimension và metric được pin trong
`model_manifest.json`. Không thay model sinh embedding mà tái sử dụng index cũ.
Caption artifacts có manifest SHA-256; `CaptionRetriever` sẽ từ chối artifact
thiếu, sai count, sai identity hoặc sai checksum thay vì âm thầm tìm kiếm trên
dữ liệu không đồng bộ.
