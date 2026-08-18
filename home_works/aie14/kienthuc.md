# Kiến thức toán học nền tảng — aie14 (math_toolkit.py)

Mục tiêu file này: note **chính xác** từng công thức, chỉ ra **tài nguyên đọc** cho từng công thức,
và nối rõ vì sao công thức đó là **xương sống của AI/ML** (không phải bài tập xa thực tế).

Bản đồ 5 kiến thức ↔ vai trò trong một mô hình học máy:

| Hàm ta implement | Kiến thức toán | Vai trò trong ML |
| --- | --- | --- |
| `cosine` | Hình học vector, tích vô hướng | So khớp embedding (RAG, search, similarity) |
| `softmax` | Hàm mũ, phân phối xác suất | Tầng output classifier; temperature lấy mẫu LLM |
| `cross_entropy` | Logarit, entropy, MLE | Hàm loss huấn luyện mọi mô hình phân loại (kể cả GPT) |
| `num_grad` | Đạo hàm, đạo hàm riêng, chuỗi Taylor | Cơ chế backpropagation; gradient checking |
| `descent` | Gradient, Hessian (curvature) | Optimizer — cách mọi mô hình "học" |

---

## 1. Cosine similarity

### Công thức chính xác

```
cos(θ) = (a · b) / (‖a‖ · ‖b‖)

a·b   = Σᵢ aᵢ bᵢ                       (tích vô hướng / dot product)
‖a‖   = √(Σᵢ aᵢ²)                      (chuẩn Euclid / L2 norm)
```

- Nguồn gốc: trong hình học, `a·b = ‖a‖·‖b‖·cos(θ)` — với θ là góc giữa 2 vector.
  Chia cả hai vế cho `‖a‖·‖b‖` là ra công thức cosine similarity.
- Thứ nguyên: cosine chỉ phụ thuộc **hướng**, không phụ thuộc **độ dài** của vector.

### Kiến thức nền cần ôn

1. **Vector trong Rⁿ**: cộng, nhân vô hướng, tọa độ.
2. **Tích vô hướng** `a·b`: cho phép "đo góc + đo chiều dài", nền tảng để định nghĩa phép chiếu.
3. **Chuẩn (norm)**: độ dài của vector; chuẩn Euclid suy từ Pythagoras.
4. **Bất đẳng thức Cauchy–Schwarz**: `|a·b| ≤ ‖a‖·‖b‖` → **chứng minh cos(θ) luôn nằm trong [−1, 1]**.
5. Hệ quả giá trị:
   - cos = 1 → cùng hướng; cos = 0 → vuông góc (orthogonal); cos = −1 → ngược hướng.

### Phép kiểm tay (phải tính trước, rồi mới assert)

Với `a = [0.9, 0.8]`, `b = [0.8, −0.9]`:

```
a·b  = 0.9·0.8 + 0.8·(−0.9) = 0.72 − 0.72 = 0
‖a‖  = √(0.81 + 0.64) = √1.45
‖b‖  = √(0.64 + 0.81) = √1.45
cos  = 0 / 1.45 = 0        → assert cosine(a, b) == 0
```

Đây là cặp **vuông góc** nên tích vô hướng triệt tiêu ngay ở tử số.

### Vì sao nó là AI/ML

- Embedding (word2vec, GloVe, BERT) biểu diễn ngữ nghĩa thành vector; độ tương đồng ngữ nghĩa
  được đo bằng cosine, không phải khoảng cách Euclid (vì khoảng cách Euclid nhạy cảm với *tần suất*
  hay *tỉ lệ co giãn* của embedding, còn nghĩa nằm ở *hướng*).
- Ứng dụng thực: semantic search, RAG retrieval, dedupe, recommendation "người dùng giống nhau".
- Chú ý kiến thức liên thông: attention trong transformer là **scaled dot-product** — đạo diễn
  cùng một kỹ thuật tích vô hướng để đo quan hệ giữa các token.

### Tài nguyên

- Wikipedia: <https://en.wikipedia.org/wiki/Cosine_similarity> , <https://en.wikipedia.org/wiki/Dot_product> ,
  <https://en.wikipedia.org/wiki/Norm_(mathematics)> , <https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality>
- 3Blue1Brown *Essence of Linear Algebra* — tập "Dot products" & "Cross products" (xem cách tích vô hướng
  liên hệ với projection): <https://www.3blue1brown.com/> (youtube: 3Blue1Brown)
- Khan Academy *Multivariable Calculus* — phần Vectors & Dot products: <https://www.khanacademy.org/math/multivariable-calculus>
- Stanford CS224n (slide word2vec dùng cosine đo similarity giữa embedding): <https://web.stanford.edu/class/cs224n/>

---

## 2. Softmax (có temperature)

### Công thức chính xác

```math
softmax(zᵢ) = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)         (T > 0)

T = 1  → softmax chuẩn
T < 1  → phân phối "sắc", dồn về đỉnh
T > 1  → phân phối "nhạt", san phẳng về uniform
```

- với exp là hàm mũ logarit tự nhiên [ e mũ (Zi / T) ]

Tính chất bắt buộc:

- `softmax(zᵢ) ≥ 0` với mọi i (vì exp luôn dương);
- `Σᵢ softmax(zᵢ) = 1` → **đúng là một phân phối xác suất**;
- Bảo toàn thứ tự: logit lớn hơn → xác suất lớn hơn; argmax được giữ nguyên.

### Kiểm ổn định số (numerical stability)

`exp(1000)` tràn (overflow) → NaN. Tận dụng `exp(a)/exp(b) = exp(a−b)` để
trừ đi max trước khi tính exp (max không đổi kết quả vì nó là hằng số chung tử-mẫu):

```
z'ᵢ = zᵢ − max(z)
softmax(zᵢ) = exp(z'ᵢ) / Σⱼ exp(z'ⱼ)
```

### Kiến thức nền cần ôn

1. **Hàm mũ `exp(x) = eˣ`**: tăng rất nhanh, luôn dương; `exp(a)/exp(b) = exp(a−b)`.
2. **Phân bố Boltzmann** (vật lý thống kê): `pᵢ ∝ exp(−Eᵢ/(kT))` — xác suất trạng thái phụ thuộc năng lượng
   và nhiệt độ. Khi ta đặt `Eᵢ = −zᵢ` và `k = 1`, đây **chính là** softmax có temperature. Vậy T đúng nghĩa là
   "nhiệt độ": T→0 đông cứng về trạng thái năng lượng thấp nhất (peak one-hot); T→∞ lẫn loạn thành uniform.
3. **Tổng biểu thức** `Σ`: đổi dấu giữa T hay logits chỉ là trước exp.
4. Quan hệ với **logistic/sigmoid**: softmax 2 lớp rút về `σ(z) = 1/(1+e^{−z})` — softmax là tổng quát hóa
   của hồi quy logistic (multinomial / softmax regression).

### Phép kiểm tay: `softmax([2, 1, 0.1])` với ba temperature

Tính nháp:

```
T=1  : exp(2), exp(1), exp(0.1) = 7.389, 2.718, 1.105 ; tổng = 11.212
       → [0.658, 0.242, 0.099]   (đỉnh 0.658)
T=0.5: exp(4), exp(2), exp(0.2) = 54.60, 7.389, 1.221 ; tổng = 63.21
       → [0.864, 0.117, 0.019]   (đỉnh nhô lên — "dồn đỉnh")
T=2  : exp(1), exp(0.5), exp(0.05) = 2.718, 1.649, 1.051 ; tổng = 5.418
       → [0.502, 0.304, 0.194]   (san phẳng hơn)
```

Giải thích một câu: temperature `T` co giãn logits *trước* khi đi qua exp — T nhỏ khiến hiệu
giữa các logits được khuếch đại (exp nhạy hơn) nên đỉnh nhô lên; T lớn khiến mọi trị gần về nhau
nên phân phối tiến về uniform. Assert: tổng mỗi dòng ≈ 1.

### Vì sao nó là AI/ML

- Tầng **output của mọi classifier**: CNN/ResNet, transformer LM đều kết thúc bằng linear → softmax
  để đọc xác suất từng lớp.
- **Sampling LLM**: khi sinh văn bản, ta lấy mẫu từ `softmax(logits/T)` — T bé → câu quyết đoán/đúng nghĩa
  đen; T lớn → sáng tạo/loạn hơn. Đây chính là tham số `temperature` trong các API LLM.
- **Attention trong transformer**: `softmax(QKᵀ/√dₖ)` — trọng số chú ý cũng là một softmax
  (đây chính là lý do `softmax` kém ổn định số sẽ làm hỏng cả mô hình).

### Tài nguyên

- Wikipedia: <https://en.wikipedia.org/wiki/Softmax_function> , <https://en.wikipedia.org/wiki/Boltzmann_distribution> ,
  <https://en.wikipedia.org/wiki/Multinomial_logistic_regression>
- CS231n *Linear Classification* (mục Softmax classifier): <https://cs231n.github.io/linear-classify/>
- Deep Learning book chương 6.2.2.3 "Softmax Units": <https://www.deeplearningbook.org/>
- HNLLM/NLP course phần sampling — ví dụ bookmark: viết prompt hỏi LLM về "temperature trong LLM"
  sau khi đã hiểu công thức, rồi đoán trước kết quả bằng tay.

---

## 3. Cross-entropy

### Công thức chính xác

```
Cross-entropy: H(P, Q) = − Σᵢ P(i) · log Q(i)

Trường hợp bài tập (P one-hot tại lớp đúng): với y_true = 1 tại lớp c,
H = −log Q(c)   →   cross_entropy(p) = −np.log(p)
```

- `−log(p)` gọi là **information content / surprise** (lượng "ngạc nhiên"): sự kiện hiếm
  (p nhỏ) mang lượng tin lớn. Ví dụ `−log 0.01 ≈ 4.605` (rất "ngạc nhiên"), `−log 0.99 ≈ 0.010` (gần như chắc chắn).
- **Entropy**: `H(P) = −Σ pᵢ log pᵢ` = lượng tin trung bình của biến ngẫu nhiên P (độ bất định).
- **KL divergence**: `KL(P‖Q) = Σ pᵢ log(pᵢ/qᵢ)`. Mối liên hệ vàng:

  ```
  H(P, Q) = H(P) + KL(P‖Q)
  ```

  Vì `H(P)` không phụ thuộc Q nên **minimize cross-entropy ≡ minimize KL** → kéo Q về gần P.

### Kiến thức nền cần ôn (bảng log) — bảng lý thuyết để assert

```
ln(0.9)  ≈ −0.105
ln(0.5)  ≈ −0.693
ln(0.01) ≈ −4.605
```

Cần ôn: định nghĩa logarit (log là hàm ngược của exp), quy tắc `log(ab) = log a + log b`
(đây là lý do *tích* xác suất của một chuỗi token trở thành *tổng* log → cross-entropy của cả câu
được tính bằng tổng từng token).

### Vì sao nó là AI/ML

- **Hàm loss chuẩn của phân loại** và của **dự đoán token kế tiếp trong GPT**: huấn luyện chính là
  tối thiểu hóa cross-entropy giữa xác suất dự đoán và nhãn one-hot thực tế.
- **Liên hệ với Maximum Likelihood Estimation (MLE)**: minimize cross-entropy = maximize
  log-likelihood của dữ liệu. "Học" = làm cho mô hình không "ngạc nhiên" về dữ liệu đúng nữa.
- Trong code thực: `nn.CrossEntropyLoss` trong PyTorch gộp `log_softmax + NLL` một lần — hiểu được
  công thức này là hiểu cách mọi framework đo loss.
- BCE (binary) là trường hợp riêng: `−[y log p + (1−y) log(1−p)]`.

### Tài nguyên

- Wikipedia: <https://en.wikipedia.org/wiki/Cross_entropy> , <https://en.wikipedia.org/wiki/Entropy_(information_theory)> ,
  <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence> ,
  <https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>
- 3Blue1Brown *Essence of Information Theory* (giải thích trực giác "lượng tin" bằng việc chơi 20-questions).
- Deep Learning book chương 3 (Probability & Information Theory): <https://www.deeplearningbook.org/>
- CS231n nhánh *Linear classification → Softmax loss*: <https://cs231n.github.io/linear-classify/>

---

## 4. Đạo hàm số `num_grad` (central difference)

### Công thức chính xác

```
Định nghĩa giới hạn:  f'(x) = lim_{h→0} (f(x+h) − f(x)) / h

Forward difference:  (f(x+h) − f(x)) / h              → sai số O(h)  (xấu hơn)
Central difference:  (f(x+h) − f(x−h)) / (2h)         → sai số O(h²) (dùng trong bài)
```

Vì sao central difference chính xác hơn (chuỗi **Taylor**):

```
f(x+h) = f(x) + f'(x)h +  ½f''(x)h² + ⅙f'''(x)h³ + ...
f(x−h) = f(x) − f'(x)h +  ½f''(x)h² − ⅙f'''(x)h³ + ...

Trừ hai vế: f(x+h) − f(x−h) = 2f'(x)h + ⅓f'''(x)h³ + ...
Chia 2h   : = f'(x) + ⅙f'''(x)h² + ...   → số hạng sai h² bị triệt tiêu
```

→ với h = 1e-5, sai số cỡ (1e-5)² = 1e-10, cực chính xác.

### Kiến thức nền cần ôn

1. **Đạo hàm = hệ số góc tiếp tuyến**, định nghĩa bằng giới hạn.
2. **Đạo hàm riêng** `∂f/∂xᵢ`: giữ mọi biến khác cố định, chỉ đạo hàm theo `xᵢ`.
   Quy tắc vàng của bài tập: **nhích từng chiều một** — copy vector, chỉ cộng h vào chiều i,
   tính f, trả về — mỗi chiều một "cảm biến" riêng.
3. **Gradient**: `∇f(x) = (∂f/∂x₁, …, ∂f/∂xₙ)` — vector chỉ **hướng tăng nhanh nhất** của f tại x.
4. **Quy tắc chuỗi (chain rule)**: `∂L/∂x = ∂L/∂y · ∂y/∂x` — nền tảng của backpropagation ở ML.

### Phép kiểm tay

`f(x,y) = x² + y²` tại `(3, 4)`:

```
∂f/∂x = 2x = 6 ;  ∂f/∂y = 2y = 8
→ gradient chính xác: ∇f = [6, 8]
→ num_grad(f, [3,4]) phải ≈ [6, 8]  (sai số nhỏ, dùng np.allclose)
```

### Vì sao nó là AI/ML

- **Backpropagation** chính là tính gradient của loss theo mọi tham số bằng đạo hàm + chain rule;
  `num_grad` là **gradient check**: cách chuẩn để kiểm tra mạng tự viết có tính gradient đúng không
  (bài tập kinh điển CS231n).
- Ý nghĩa trực giác: gradient cho biết "nên đẩy tham số theo hướng nào để loss giảm" — mọi thuật toán
  học (SGD, Adam) đều cần giá trị này.

### Tài nguyên

- Wikipedia: <https://en.wikipedia.org/wiki/Numerical_differentiation> , <https://en.wikipedia.org/wiki/Partial_derivative> ,
  <https://en.wikipedia.org/wiki/Gradient> , <https://en.wikipedia.org/wiki/Taylor_series>
- CS231n *Backpropagation, Intuitions*: <https://cs231n.github.io/optimization-2/>
- 3Blue1Brown *Essence of Calculus* (đạo hàm + Taylor series trực quan); *Neural Networks* tập 3
  "What is backpropagation really doing?".

---

## 5. Gradient descent

### Công thức chính xác

```
x_{k+1} = x_k − η · ∇f(x_k)        với η > 0 là learning rate

Vì −∇f là hướng tăng chậm nhất (đi ngược hướng tăng nhanh nhất nên f giảm).
Xấp xỉ bậc nhất:  f(x − η∇f) ≈ f(x) − η·‖∇f‖²  <  f(x)  khi η đủ nhỏ.
```

Với `f(x,y) = x² + y²` (paraboloid), gradient `∇f = (2x, 2y)` — tuyến tính, nên phân tích chính xác:

```
x_{k+1} = x_k − 2η x_k = (1 − 2η)·x_k     (từng chiều độc lập)

f = 25 tại (3,4) → sau k bước: vị trí (3·qᵏ, 4·qᵏ) với q = 1 − 2η;   f_k = 25·q^(2k)
```

### Hội tụ / phân kỳ phụ thuộc η (đây là phân tích cần hiểu)

| η | q = 1 − 2η | |q| | Hành vi | Tên thí nghiệm |
|---|---|---|---|---|
| 0.2 | 0.6 | 0.6 < 1 | Hội tụ nhanh, không dao động | lr hợp lý |
| 0.01 | 0.98 | 0.98 < 1 | Co cực chậm, "rùa bò" | lr quá nhỏ |
| 1.1 | −1.2 | 1.2 > 1 | Đổi dấu liên tục, độ lớn ×1.2 mỗi bước → văng xa dần | lr quá lớn |

- Điều kiện hội tụ của bậc 2: `|1 − 2η| < 1` ⇔ `0 < η < 1`. Sau đó chia cho **curvature**:
  với Hessian `∇²f = 2I`, eigenvalue `λ_max = 2`, điều kiện tổng quát `η < 2/λ_max`.
- **Bước giảm loss mạnh nhất là bước 1**: f từ 25 → 9 (giảm 16, ~64%) vì gradient tỉ lệ với ‖x‖ —
  càng xa điểm cực tiểu bước càng dài; về sau grad nhỏ dần nên co chậm lại.

### Phép kiểm tay lr = 0.2, 10 bước từ (3, 4)

```
k     vị trí                        f = x²+y²
0     (3.000, 4.000)                25.000     ← gradient (6,8) lớn nhất, giảm mạnh nhất
1     (1.800, 2.400)                9.000
2     (1.080, 1.440)                3.240
...
10    (3·0.6¹⁰, 4·0.6¹⁰) ≈ (0.018, 0.024)   f ≈ 0.0009  → co về (0,0) rõ rệt
```

### Kiến thức nền cần ôn

1. Gradient đã ôn ở mục 4; cần thêm ý: gradient tay lái hướng, learning rate quyết định cỡ bước.
2. **Hessian / curvature** `∇²f`: đo "độ cong" — học rate nên tỉ lệ nghịch với độ cong
   (lý do phải thử lr, dùng schedule, hay optimizer thích nghi như Adam).
3. Khái niệm **local minimum vs global minimum**, **convexity** (`x²+y²` là lồi → min duy nhất).

### Vì sao nó là AI/ML

- Đây là công thức mà **mọi mạng thần kinh học**: model có hàng triệu tham số θ, và cập nhật
  `θ ← θ − η∇Loss` chính là trái tim của SGD/Adam trong PyTorch/TensorFlow.
- "Học" không có gì thần bí: chỉ là tối thiểu hóa một hàm loss bằng cách đi xuống đồi —
  hiểu η để biết vì sao "lr quá cao không train được" hoặc "lr quá thấp học mãi không tiến".
- Adam về bản chất là GD thêm ước lượng moment — hiểu GD gốc là bước đệm bắt buộc.

### Tài nguyên

- Wikipedia: <https://en.wikipedia.org/wiki/Gradient_descent> , <https://en.wikipedia.org/wiki/Hessian_matrix> ,
  <https://en.wikipedia.org/wiki/Stochastic_gradient_descent> , <https://en.wikipedia.org/wiki/Mathematical_optimization>
- 3Blue1Brown *Neural Networks* tập 2 "Gradient descent, how neural networks learn" — video trực giác nhất về GD.
- CS231n *Optimization*: <https://cs231n.github.io/optimization-1/>
- Distill.pub *Why momentum really works* (đi sâu hơn): <https://distill.pub/2017/momentum/>
- Deep Learning book chương 4 (Numerical Computation) & 8 (Optimization): <https://www.deeplearningbook.org/>

---

## Tổng kết mạch liên kết (đọc lại sau khi xong 5 mục)

Một mô hình ML điển hình dùng **cả 4 hàm này trong một pipeline**:

1. Embedding/dữ liệu được biểu diễn thành vector → **cosine** để so sánh (retrieval, tương đồng).
2. Mô hình ra logits → **softmax** chuyển thành xác suất phân phối.
3. So phân phối dự đoán với nhãn thật bằng **cross-entropy** để ra con số loss.
4. Loss là hàm số của tham số → tính gradient bằng đạo hàm (**num_grad** là cách kiểm chứng
   gradient giải tích của backprop).
5. Lấy gradient nhân learning rate mà **trừ vào tham số** → tham số mới → lặp lại (gradient descent).

Đọc xong file này theo thứ tự 1 → 5 trong khi tự tính tay từng ví dụ gần giống các phép kiểm ở trên,
rồi viết `math_toolkit.py` và assert từng hàm — đó chính là cách toán "vào người".
