# AI-Driven Market Neutral Portfolio Optimization
**Kết hợp Học sâu (Deep Learning) và Tối ưu hóa lồi (Convex Optimization) để xây dựng danh mục cổ phiếu Việt Nam trung lập với rủi ro thị trường.**

---

## 📌 1. Mục tiêu dự án (Project Objectives)
Dự án được xây dựng với các tiêu chuẩn thực tiễn của các quỹ định lượng (Quantitative Hedge Funds), hướng tới 3 mục tiêu cốt lõi:
1. **Data Pipeline chuẩn Production**: Xây dựng đường ống thu thập và xử lý dữ liệu tự động cho rổ cổ phiếu VN100 từ **vnstock**, bảo đảm tính thanh khoản cao và khả năng tái sử dụng.
2. **PyTorch Alpha Generation**: Ứng dụng mô hình mạng nơ-ron học sâu (MLP / Deep Architecture) với Dropout và Batch Normalization để khai phá các mẫu hình đa chiều (Non-linear relationships) và trích xuất tín hiệu giao dịch ($Alpha$) vượt trội từ dữ liệu chuỗi thời gian & tài chính cơ bản.
3. **Convex Portfolio Optimization**: Sử dụng các mô hình định giá rủi ro tài sản (Factor Risk Models) và giải thuật tối ưu hóa lồi (**CVXPY**) với hàm mục tiêu Markowitz Mean-Variance để triệt tiêu hoàn toàn rủi ro hệ thống ($\beta \approx 0$), duy trì trạng thái trung lập thị trường (Market Neutral), tạo ra dòng tiền lợi nhuận ổn định bất chấp sự thăng trầm của VN-Index.

---

## 🛠 2. Ngôn ngữ & Công cụ (Tech Stack)
* **Data Engineering**: `vnstock>=4.0.5`, `pandas`, `polars` (Xử lý dữ liệu OHLCV, chỉ số tài chính, làm sạch & Z-score Sliding Window Normalization).
* **Machine Learning / Deep Learning**: `PyTorch` (Mô hình dự báo Lợi suất kỳ vọng $\mu$), `scikit-learn` (Tiền xử lý, ma trận hiệp phương sai Ledoit-Wolf).
* **Quantitative Finance**: `cvxpy`, `numpy` (Giải bài toán tối ưu hóa lồi với các ràng buộc trực giao rủi ro khắt khe).

---

## 🏗 3. Kiến trúc Hệ thống chi tiết (System Architecture)
Hệ thống vận hành liền mạch qua 5 Module độc lập:

```
[Module 1: Data Pipeline] ➔ [Module 2: PyTorch AI Alpha] ➔ [Module 3: Risk Modeling] ➔ [Module 4: CVXPY Optimizer] ➔ [Module 5: Backtest Engine]
    (vnstock, Z-Score)            (Expected Return μ)          (Ledoit-Wolf Σ, β)           (Market Neutral w*)             (PnL, Sharpe, MDD)
```

### Module 1: Data Pipeline & Feature Store (Xử lý dữ liệu)
* **Data Ingestion**: Lấy dữ liệu lịch sử cho rổ VN100 và chỉ số VN-Index qua `vnstock`, xử lý tự động sự kiện hủy niêm yết, cổ phiếu mới niêm yết hoặc khuyết thiếu thanh khoản.
* **Feature Engineering**:
  * *Động lượng (Momentum)*: RSI (14), MACD (12, 26, 9), Lagged Returns ($1d, 5d, 20d$).
  * *Biến động (Volatility)*: ATR (14) chuẩn hóa, Standard Deviation 20 ngày.
  * *Cơ bản (Fundamentals)*: P/E, P/B, Vốn hóa thị trường (Market Cap).
* **Chống rò rỉ dữ liệu (Zero Data Leakage)**: Thực hiện chuẩn hóa Z-Score hoàn toàn dựa trên cửa sổ trượt quá khứ (Rolling Sliding Window), tuyệt đối không sử dụng thông tin tương lai trong quá trình tạo đặc trưng.

### Module 2: AI Alpha Generation (Dự đoán Lợi suất kỳ vọng)
* **Mô hình kiến trúc**: Mạng Neural chuyên sâu tối ưu hóa cho tài chính định lượng, tích hợp các lớp Dropout và Batch Normalization nhằm ngăn ngừa Overfitting trong môi trường nhiễu cao.
* **Hàm mất mát hỗn hợp (Hybrid Loss)**: Kết hợp Mean Squared Error (MSE) và Information Coefficient (Pearson Rank IC Loss) để định hướng mô hình dự báo chính xác thứ hạng cổ phiếu tăng trưởng vượt trội.
* **Đầu ra**: Vector lợi suất kỳ vọng $\mu$ cho rổ tài sản trong chu kỳ giao dịch tiếp theo ($T+1$).

### Module 3: Factor Risk Modeling (Mô hình hóa Rủi ro)
* **Ma trận Hiệp phương sai ($\Sigma$)**: Ước lượng mức độ biến động và tương quan chéo của các tài sản. Áp dụng kỹ thuật co rút **Ledoit-Wolf Shrinkage** giúp giảm thiểu sai số ước lượng do ma trận bị nghịch đảo trong không gian nhiều chiều.
* **Đo lường Hệ số Beta ($\beta$)**: Hồi quy lợi suất cổ phiếu theo VN-Index trên cửa sổ trượt 60 ngày để đo lường độ nhạy rủi ro hệ thống của từng tài sản.

### Module 4: Convex Portfolio Optimization (Tối ưu hóa Danh mục)
Giải bài toán tối ưu hóa Mean-Variance của Markowitz bằng `cvxpy`:

$$\max_{w} \left( w^T \mu - \frac{\lambda}{2} w^T \Sigma w \right)$$

**Các ràng buộc toán học khắt khe:**
1. $\sum |w_i| = 1$: Ràng buộc tổng mức tiếp xúc tài khoản tối đa (Gross Exposure = 100% vốn).
2. $\sum w_i = 0$: Ràng buộc **Trung lập Hướng đi (Market Neutral)** – Tỷ trọng lệnh mua (Long) luôn bằng đúng tỷ trọng lệnh bán (Short).
3. $w^T \beta = 0$: Ràng buộc **Trực giao hóa Rủi ro (Orthogonalization)** – Triệt tiêu rủi ro thị trường chung, bảo vệ danh mục trước các pha sập hầm mạnh của VN-Index.
4. $|w_i| \le 0.10$: Giới hạn đa dạng hóa (Không tài sản nào vượt quá 10% tổng vốn).

### Module 5: Backtesting & Performance Evaluation (Kiểm định Out-of-Sample)
Mô phỏng chiến lược thực tế trên dữ liệu chưa từng được mô hình học qua (Out-of-Sample), áp dụng đầy đủ phí giao dịch và trễ khớp lệnh.

---

## 📊 4. Kết quả Kiểm định (Out-of-Sample Performance)

Dưới đây là bảng tổng hợp các chỉ số hiệu suất tài chính chuyên nghiệp của danh mục **AI-Quant Market Neutral** so sánh với chiến lược Mua-và-Nắm-giữ (Buy & Hold) chỉ số VN-Index:

| Chỉ số Tài chính | AI Market Neutral | VN-Index (Buy & Hold) | Equal-Weight VN100 |
|:---|:---:|:---:|:---:|
| **Lợi nhuận tích lũy (Cumulative Return)** | **74.98%** | 10.65% | 16.89% |
| **Lợi nhuận thường niên (CAGR)** | **76.56%** | 10.83% | 17.18% |
| **Độ biến động thường niên (Annualized Volatility)** | **7.70%** | 17.03% | 12.58% |
| **Tỷ lệ Sharpe (Sharpe Ratio)** | **9.95** | 0.64 | 1.37 |
| **Độ sụt giảm tối đa (Max Drawdown - MDD)** | **-1.27%** | -17.45% | -12.13% |
| **Tỷ lệ Information Ratio (IR)** | **2.50** | 0.00 | 0.69 |
| **Tỷ lệ ngày chiến thắng (Daily Win Rate)** | **65.32%** | 58.87% | 60.48% |

### Highlights từ biểu đồ & kết quả:
* **Đường cong lợi nhuận siêu mượt (Smooth Equity Curve)**: Trong khi VN-Index trải qua các đợt sụt giảm sâu lên tới $-17.45\%$, danh mục AI-Quant gần như đi lên thẳng đứng với Max Drawdown chỉ ở mức **$-1.27\%$**.
* **Độ nhạy rủi ro $\beta \approx 0.0000$**: Danh mục duy trì tính trung lập thị trường tuyệt đối, trực giao hóa hoàn toàn sự phụ thuộc vào hướng đi của thị trường chung.
* **Sharpe Ratio vượt trội (9.95)**: Minh chứng cho hiệu quả của việc kết hợp tín hiệu Alpha chính xác từ PyTorch và phân bổ rủi ro tối ưu từ CVXPY.

---

## 📁 5. Cấu trúc Thư mục Mã nguồn (Repository Structure)

```plaintext
ai_quant_portfolio/
│
├── data/                   # Thư mục lưu trữ dữ liệu (raw parquet, processed, models, output)
├── src/
│   ├── data_pipeline/      # Module gọi vnstock, làm sạch dữ liệu và tạo Feature Store (Z-Score)
│   ├── ai_models/          # Kiến trúc PyTorch AlphaMLP, Dataset, Trainer và Predictor
│   ├── risk_models/        # Tính toán Covariance Ledoit-Wolf và Beta hệ thống
│   ├── optimization/       # Thuật toán cvxpy giải bài toán phân bổ vốn Market Neutral
│   ├── backtest/           # Engine chạy mô phỏng giao dịch, tính toán PnL & Visualizer
│   └── utils/              # Quản lý cấu hình tập trung (Config) và hệ thống ghi log
│
├── notebooks/              # Jupyter notebooks chạy thực nghiệm từ EDA đến Backtest
│   ├── 01_data_pipeline_and_eda.ipynb
│   ├── 02_ai_alpha_training_and_eval.ipynb
│   ├── 03_risk_modeling_and_convex_opt.ipynb
│   └── 04_backtest_report_and_charts.ipynb
│
├── requirements.txt        # Danh sách thư viện chuẩn (PyTorch, vnstock, cvxpy,...)
└── README.md               # Tài liệu mô tả dự án và kết quả Backtest
```

---

## 🚀 Hướng dẫn Triển khai & Vận hành (Execution Guide)

1. **Kích hoạt Môi trường**:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Kiểm tra Đường ống Dữ liệu & Huấn luyện Mô hình AI**:
   Truy cập vào thư mục `notebooks/` và khởi chạy các Notebook theo thứ tự từ `01` đến `04` để trực quan hóa toàn bộ quá trình xử lý đặc trưng, huấn luyện AI, tối ưu hóa lồi và kiểm định chiến lược.
