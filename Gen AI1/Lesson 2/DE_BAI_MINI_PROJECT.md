# 📋 ĐỀ BÀI MINI PROJECT: FullStack Credit Score Prediction App

> **Tác giả**: Nguyễn Quốc Huy (Rinez) — VTI Academy

---

## Outline

1. [Tổng quan đề bài](#1-tổng-quan-đề-bài)
2. [Yêu cầu kỹ thuật bắt buộc](#2-yêu-cầu-kỹ-thuật-bắt-buộc)
3. [Yêu cầu chức năng chi tiết](#3-yêu-cầu-chức-năng-chi-tiết)
4. [Sản phẩm nộp bài](#4-sản-phẩm-nộp-bài)
5. [Tiêu chí chấm điểm](#5-tiêu-chí-chấm-điểm)
6. [Tài liệu tham khảo](#6-tài-liệu-tham-khảo)
7. [Hình thức nộp](#hình-thức-nộp)
8. [Gợi ý tính năng](#8-gợi-ý-tính-năng)

---

## 1. Tổng quan đề bài

### Mô tả dự án

Xây dựng một ứng dụng web **fullstack** hoàn chỉnh cho bài toán **Credit Score Prediction** (Dự đoán điểm tín dụng phê duyệt khoản vay). Ứng dụng cho phép người dùng nhập thông tin hồ sơ vay, hệ thống sẽ sử dụng mô hình Machine Learning để đánh giá và trả về kết quả phê duyệt.

### Mục tiêu

- Áp dụng kiến thức ML để train và export model phục vụ production
- Xây dựng REST API backend bằng Python
- Xây dựng giao diện frontend hiện đại
- Kết nối database lưu trữ dữ liệu
- Deploy toàn bộ hệ thống lên cloud
- Hiểu quy trình phát triển phần mềm end-to-end

---

## 2. Yêu cầu kỹ thuật bắt buộc

### 2.1. Tech Stack quy định

| Layer | Công nghệ | Hosting |
|-------|-----------|---------|
| **ML Model** | XGBoost (hoặc scikit-learn) | File `.json` / `.pkl` trong repo |
| **Backend** | FastAPI (Python) | Render.com |
| **Frontend** | Next.js 14+ (React) | Vercel |
| **Database** | PostgreSQL | Supabase |
| **Version Control** | Git + GitHub | GitHub |

### 2.2. Kiến trúc bắt buộc

```
[User Browser]
      ↓ HTTPS
[Frontend - Vercel]
      ↓ REST API (JSON)
[Backend - Render]
      ↓ load model          ↓ SQL
[ML Model file]        [Supabase PostgreSQL]
```

---

## 3. Yêu cầu chức năng chi tiết

### 3.1. Machine Learning (Model)

| # | Yêu cầu | Mức độ |
|---|---------|--------|
| M1 | Sử dụng dataset `credit_data.csv` (được cung cấp) hoặc tự tìm dataset tương đương | Bắt buộc |
| M2 | Thực hiện EDA: phân tích phân bố features, kiểm tra missing values, correlation | Bắt buộc |
| M3 | Preprocessing: chia train/test (80/20), xử lý missing values nếu có | Bắt buộc |
| M4 | Train model XGBoost (hoặc model khác) cho bài toán classification | Bắt buộc |
| M5 | Đánh giá model: Accuracy, ROC AUC, Confusion Matrix | Bắt buộc |
| M6 | Export model ra file (`.json` hoặc `.pkl`) để backend load | Bắt buộc |
| M7 | Vẽ biểu đồ Feature Importance | Nâng cao |
| M8 | So sánh với model khác (Random Forest, Logistic Regression) | Bonus |

**Output yêu cầu**: Notebook `.ipynb` + file model exported

### 3.2. Backend API

| # | Yêu cầu | Mức độ |
|---|---------|--------|
| B1 | Tạo FastAPI app với Swagger UI tự động (`/docs`) | Bắt buộc |
| B2 | Endpoint `GET /health` — health check (trả trạng thái model + DB) | Bắt buộc |
| B3 | Endpoint `POST /predict` — nhận input JSON, predict bằng model, trả kết quả | Bắt buộc |
| B4 | Validate input bằng Pydantic (type, min/max cho từng field) | Bắt buộc |
| B5 | Response phải bao gồm: `approval_score`, `approved` (bool), `risk_level`, `recommendation` | Bắt buộc |
| B6 | Kết nối Supabase để lưu kết quả mỗi lần predict | Bắt buộc |
| B7 | Endpoint `GET /applications` — lấy lịch sử (có pagination) | Bắt buộc |
| B8 | CORS middleware cho phép frontend gọi API | Bắt buộc |
| B9 | Graceful degradation: API vẫn hoạt động khi DB không kết nối được | Nâng cao |
| B10 | Endpoint `GET /applications/{id}` — chi tiết 1 record | Nâng cao |
| B11 | Endpoint `GET /model-info` — trả thông tin model đang dùng | Bonus |

**Input fields bắt buộc cho `/predict`** (tối thiểu 10 fields):

| Field | Type | Mô tả |
|-------|------|--------|
| `income` | float | Thu nhập hàng năm (USD) |
| `age` | int | Tuổi |
| `employment_years` | int | Số năm đi làm |
| `loan_amount` | float | Số tiền vay |
| `loan_term` | int | Kỳ hạn vay (tháng) |
| `credit_history_length` | int | Thời gian có lịch sử tín dụng (năm) |
| `num_credit_lines` | int | Số khoản tín dụng đang có |
| `num_delinquencies` | int | Số lần trễ hạn thanh toán |
| `debt_to_income_ratio` | float | Tỷ lệ nợ/thu nhập |
| `savings_balance` | float | Số dư tiết kiệm |

> Có thể thêm fields khác: `property_value`, `education_level`, `employment_type`, v.v.

### 3.3. Database

| # | Yêu cầu | Mức độ |
|---|---------|--------|
| D1 | Tạo table `applications` lưu: input_data, score, approved, risk_level, timestamp | Bắt buộc |
| D2 | Dùng kiểu JSONB cho `input_data` (lưu toàn bộ input linh hoạt) | Bắt buộc |
| D3 | Primary key UUID tự tạo | Bắt buộc |
| D4 | Bật Row Level Security (RLS) + policy cho anon user | Bắt buộc |
| D5 | Có index trên `created_at` để query lịch sử nhanh | Nâng cao |
| D6 | Thêm trigger tự update `updated_at` | Bonus |

### 3.4. Frontend

| # | Yêu cầu | Mức độ |
|---|---------|--------|
| F1 | Trang chủ `/` — giới thiệu ứng dụng | Bắt buộc |
| F2 | Trang `/apply` — form nhập đơn vay (tối thiểu 10 fields) | Bắt buộc |
| F3 | Hiển thị kết quả sau khi submit: score, approved/rejected, risk level | Bắt buộc |
| F4 | Trang `/history` — bảng lịch sử các đơn đã submit | Bắt buộc |
| F5 | Responsive design (hiển thị tốt trên mobile + desktop) | Bắt buộc |
| F6 | Loading state khi chờ API respond | Bắt buộc |
| F7 | Error handling: hiển thị thông báo khi API lỗi | Bắt buộc |
| F8 | Form validation: không cho submit nếu thiếu thông tin | Bắt buộc |
| F9 | Hiển thị recommendation text từ backend | Nâng cao |
| F10 | Biểu đồ hoặc gauge hiển thị score trực quan | Nâng cao |
| F11 | Quick-fill mẫu data (nút điền nhanh data demo) | Nâng cao |
| F12 | Thống kê tổng hợp trên trang history (tổng đơn, % approved, avg score) | Bonus |

### 3.5. Deployment

| # | Yêu cầu | Mức độ |
|---|---------|--------|
| P1 | Backend deploy trên Render.com | Bắt buộc |
| P2 | Frontend deploy trên Vercel | Bắt buộc |
| P3 | Database trên Supabase (đã setup) | Bắt buộc |
| P4 | Environment variables cấu hình đúng trên cả Render + Vercel | Bắt buộc |
| P5 | CORS cấu hình đúng (backend allow domain frontend) | Bắt buộc |
| P6 | Auto-deploy khi push code lên GitHub | Nâng cao |

---

## 4. Sản phẩm nộp bài

### 4.1. Source code (GitHub)

Nộp **2 GitHub repositories** (hoặc 1 monorepo):

| Repo | Nội dung |
|------|---------|
| `credit-score-backend` | Code backend + model files + notebook |
| `credit-score-frontend` | Code frontend Next.js |

**Mỗi repo phải có:**
- `README.md` — hướng dẫn cài đặt và chạy local
- `.gitignore` — không push `node_modules/`, `venv/`, `.env`, `__pycache__/`
- Code sạch, có comment ở những chỗ quan trọng

### 4.2. URLs production

| Service | URL |
|---------|-----|
| Frontend (Vercel) | `https://your-app.vercel.app` |
| Backend API (Render) | `https://your-api.onrender.com` |
| Swagger UI | `https://your-api.onrender.com/docs` |

### 4.3. Demo

- Trình bày demo ứng dụng hoạt động trên production (Vercel URL)
- Thao tác: nhập đơn → xem kết quả → xem lịch sử
- Giải thích luồng hoạt động: Frontend → Backend → Model → Database

---

## 5. Tiêu chí chấm điểm

### Bảng điểm (Tổng: 100 điểm)

| Hạng mục | Điểm | Chi tiết |
|---------|------|---------|
| **ML Model** | **15** | |
| - Train model chạy được, export thành công | 8 | Notebook chạy đầy đủ, model predict được |
| - EDA + Evaluation đầy đủ | 5 | Có biểu đồ, metrics rõ ràng |
| - So sánh models / Feature Importance | 2 | Bonus |
| **Backend API** | **25** | |
| - `/predict` hoạt động đúng | 10 | Nhận input → trả score, approved, risk_level |
| - Pydantic validation | 5 | Input sai → trả lỗi rõ ràng |
| - `/health` + `/applications` | 5 | Health check + lịch sử hoạt động |
| - Kết nối Supabase lưu data | 5 | Mỗi predict lưu vào DB thành công |
| **Database** | **10** | |
| - Table schema đúng + RLS | 6 | Table tạo đúng, RLS bật, policies đúng |
| - Data lưu đúng + query được | 4 | Verify data trong Supabase Dashboard |
| **Frontend** | **25** | |
| - Form nhập đơn (≥10 fields) | 8 | Form hiển thị đúng, đủ fields |
| - Hiển thị kết quả score | 7 | Score, approved/rejected, risk level |
| - Trang lịch sử | 5 | Bảng hiển thị đúng data |
| - UI/UX + Responsive | 5 | Giao diện sạch, dùng được trên mobile |
| **Deployment** | **15** | |
| - Backend live trên Render | 5 | URL hoạt động, `/docs` mở được |
| - Frontend live trên Vercel | 5 | URL hoạt động, 3 trang hiển thị |
| - End-to-end test thành công | 5 | Submit trên Vercel → Data lưu vào Supabase |
| **Code Quality** | **10** | |
| - README.md rõ ràng | 3 | Hướng dẫn setup local đầy đủ |
| - Code có cấu trúc, comment | 4 | Tách file hợp lý, comment chỗ quan trọng |
| - .gitignore đúng, không push secrets | 3 | Không push `.env`, `node_modules`, `venv` |

---

## 6. Tài liệu tham khảo

### Tài liệu dự án (đã cung cấp sẵn)

| File | Mô tả |
|------|--------|
| `FULL_SETUP_GUIDE.md` | Hướng dẫn setup từ Step 1 → Step 5 |
| `Step01_Train_Model/` | Notebook mẫu + sample data |
| `Step02_Backend/` | Code backend mẫu |
| `Step03_Database/` | SQL schema + seed data |
| `Step04_Frontend/` | Code frontend mẫu |
| `Step05_Deployment/` | Hướng dẫn deploy từng platform |

### Documentation chính thức

| Công nghệ | Link |
|-----------|------|
| FastAPI | https://fastapi.tiangolo.com |
| Next.js | https://nextjs.org/docs |
| Supabase | https://supabase.com/docs |
| XGBoost | https://xgboost.readthedocs.io |
| Tailwind CSS | https://tailwindcss.com/docs |
| Render | https://docs.render.com |
| Vercel | https://vercel.com/docs |

---

### Hình thức nộp

1. **Link GitHub**: 2 repo (backend + frontend)
2. **Link production**: URL Vercel + URL Render
3. **Demo trực tiếp**: trình bày trước lớp (5-10 phút / nhóm)

---

## 8. Gợi ý tính năng

| # | Tính năng |
|---|----------|
| 1 | Thêm **authentication** (Supabase Auth: đăng ký / đăng nhập) |
| 2 | Thêm **dashboard** thống kê (biểu đồ chart.js hoặc recharts) |
| 3 | **Batch prediction**: upload file CSV → predict nhiều hồ sơ |
| 4 | **So sánh 2+ models**: cho user chọn model để predict |
| 5 | **Export PDF**: xuất kết quả ra file PDF |
| 6 | **Dark mode**: toggle giao diện sáng/tối |
| 7 | **Animation**: hiệu ứng khi hiển thị kết quả (gauge quay, score đếm lên) |
| 8 | Tự tìm **dataset thật** trên Kaggle thay vì dùng sample data |
| 9 | Viết **unit test** cho backend (pytest) |
| 10 | Thêm **CI/CD pipeline** (GitHub Actions) |

---

*Credit Score Prediction — FullStack VibeCoding Mini Project*

> **Tác giả**: Nguyễn Quốc Huy (Rinez) — VTI Academy
