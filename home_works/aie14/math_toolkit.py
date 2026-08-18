import numpy as np


def cosine(vec1: list[float], vec2: list[float]) -> float:
    """Tính cosine similarity giữa hai vector."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    # tích vô hướng của hai vector
    # SUM i -> N (v1[i] * v2[i])
    dot_product = np.dot(v1, v2)
    # độ dài của vector v1 và v2
    norm_v1 = np.linalg.norm(v1) # độ dài vector v1
    norm_v2 = np.linalg.norm(v2) # độ dài vector v2

    if norm_v1 == 0 or norm_v2 == 0:
        return 0

    return dot_product / (norm_v1 * norm_v2)

def softmax(logits: list[float], T: float = 1.0) -> list[float]:
    """Tính softmax của một vector logits với nhiệt độ T."""
    # Logits là output của mô hình AI (1 vector các giá trị)
    # ví dụ dự đoán mèo, chó, gà như sau: [2.0, 1.0, 0.1]
    # -> là Logits, là output của mô hình W.x + b  (với W là trọng số ma trận, x là input, b là bias.)
    # vì là output của mô hình nên giá trị các phần tử trong Logits có miền giá trị từ -inf đến +inf, không có giới hạn.
    # vậy làm sao để chuyển đổi các giá trị trong Logits về xác suất (probability) có miền giá trị từ 0 đến 1 và tổng các xác suất bằng 1?
    
    # bước 1: ta có ý tưởng là chia tỉ lệ tuyến tính:
    # từng phần tử S[i] trong Logits / tổng các phần tử trong Logits
    # ví dụ: Logits = [2.0, 1.0, 0.1] -> xác suất = [2.0 / (2.0 + 1.0 + 0.1), 1.0 / (2.0 + 1.0 + 0.1), 0.1 / (2.0 + 1.0 + 0.1)] = [0.645, 0.322, 0.032]
    # nhưng cách này không tốt vì nếu Logits có giá trị âm thì xác suất sẽ âm, không hợp lệ.
    # bước 2: ta có ý tưởng là dùng hàm mũ logarit (exponential) để chuyển đổi các giá trị trong Logits về miền giá trị từ 0 đến +inf. (số càng âm thì giá trị càng sát 0, số càng dương thì giá trị càng lớn)
    # ví dụ: Logits = [2.0, 1.0, 0.1] -> exp(Logits) = [exp(2.0), exp(1.0), exp(0.1)] = [7.389, 2.718, 1.105]
    # bước 3: sau khi đã có các giá trị trong miền từ 0 đến +inf, ta chia tỉ lệ tuyến tính như bước 1 để chuyển đổi về xác suất.
    # ví dụ: exp(Logits) = [7.389, 2.718, 1.105] -> xác suất = [7.389 / (7.389 + 2.718 + 1.105), 2.718 / (7.389 + 2.718 + 1.105), 1.105 / (7.389 + 2.718 + 1.105)] = [0.645, 0.322, 0.032]
    
    # đó là softmax thuần túy, vậy thêm nhiệt độ T vào mẫu số của mũ trong hàm mũ logarit để làm gì?
    # ta thêm nhiệt độ T vào mẫu số của hàm mũ logarit để điều chỉnh độ phân bố của các xác suất.
    # nếu T > 1 thì các xác suất sẽ phân bố đều hơn, nếu T < 1 thì các xác suất sẽ phân bố tập trung hơn vào các giá trị lớn trong Logits.
    # ví dụ: Logits = [2.0, 1.0, 0.1], T = 2.0 -> exp(Logits / T) = [exp(2.0 / 2.0), exp(1.0 / 2.0), exp(0.1 / 2.0)] = [2.718, 1.648, 1.051] -> xác suất = [2.718 / (2.718 + 1.648 + 1.051), 1.648 / (2.718 + 1.648 + 1.051), 1.051 / (2.718 + 1.648 + 1.051)] = [0.462, 0.280, 0.258]

    # YÊU CẦU 2 — Ổn định số: trừ max(logits) TRƯỚC khi exp để tránh tràn số.
    # Vấn đề: logits lớn (vd 1000) -> exp(1000) = inf (overflow) -> kết quả NaN.
    # Vì exp(a)/exp(b) = exp(a - b): trừ cùng hằng số max ở tử và mẫu bị triệt tiêu
    # nên kết quả toán học KHÔNG đổi, nhưng exp(z_i - max) <= exp(0) = 1
    # nên không bao giờ tràn được nữa.
    lgs = np.array(logits, dtype=float)
    lgs_shifted = lgs - np.max(lgs)

    # (z_i - max) / T luôn <= 0 kể cả khi T < 1 -> vẫn an toàn sau khi chia T.
    # công thức softmax: softmax(x_i) = exp(x_i / T) / SUM j (exp(x_j / T))
    exp_lgs = np.exp(lgs_shifted / T)

    sum_exp_lgs = np.sum(exp_lgs)
    probs = exp_lgs / sum_exp_lgs  # vector xác suất, mỗi phần tử trong (0, 1)

    # YÊU CẦU 2 — assert tổng xác suất ≈ 1 (tổng một phân phối xác suất phải bằng 1)
    np.testing.assert_allclose(np.sum(probs), 1.0)
    return probs.tolist()


def cross_entropy(p_dung: list[float]) -> list[float]:
    """Hàm 3 — cross_entropy: -log(p) cho từng xác suất p của lớp ĐÚNG.

    Nếu nhãn đúng là one-hot (y_c = 1, còn lại 0), công thức đầy đủ
    H(y, p) = -SUM_i y_i * log(p_i) rút gọn thành đúng -log(p_dung).
    np.log() là logarit tự nhiên (cơ số e) — khớp với bảng lý thuyết.
    """
    probs = np.array(p_dung, dtype=float)
    # cơ chế của cross entropy: nếu p_dung = 1 (đúng hoàn toàn) -> -log(1) = 0 (không loss); nếu p_dung = 0 (sai hoàn toàn) -> -log(0) = +inf (loss vô hạn)
    # vì sao nhãn đúng là one-hot? vì trong bài toán phân loại, mỗi mẫu chỉ thuộc về một lớp duy nhất, nên xác suất của lớp đúng sẽ là 1, còn các lớp khác sẽ là 0. Khi tính cross-entropy, ta chỉ quan tâm đến xác suất của lớp đúng, nên công thức rút gọn thành -log(p_dung).
    # công thức gốc của cross-entropy: H(y, p) = -SUM_i y_i * log(p_i) với y_i là nhãn đúng (one-hot), p_i là xác suất dự đoán của lớp i. Khi y_c = 1 (lớp đúng), các y_i khác = 0, nên chỉ còn lại -log(p_dung).
    # -log(p): p = 1 -> 0 (không ngạc nhiên); p -> 0 -> +inf (sai hoàn toàn)
    # mục tiêu của cross-entropy là đo độ "ngạc nhiên" của mô hình khi dự đoán nhãn đúng. Nếu mô hình dự đoán xác suất cao cho lớp đúng (p_dung gần 1), loss sẽ thấp (gần 0). Nếu mô hình dự đoán xác suất thấp cho lớp đúng (p_dung gần 0), loss sẽ cao (gần +inf).
    
    # mình cho ví dụ mô hình nhận diện mèo, chó, gà
    # với logits = [2.0, 1.0, 0.1] -> softmax = [0.645, 0.322, 0.032].
    # Nếu nhãn đúng là mèo (lớp 0), p_dung = 0.645 -> -log(0.645) = 0.438 (loss thấp).
    # Nếu nhãn đúng là gà (lớp 2), p_dung = 0.032 -> -log(0.032) = 3.44 (loss cao).
    # Điều này phản ánh rằng mô hình dự đoán sai lớp đúng sẽ bị phạt nặng hơn.
    
    # vì sao công thức cross-entropy triệt tiêu hết các nhãn đánh dấu là 0?
    # Công thức gốc của cross-entropy: H(y, p) = -SUM_i y_i * log(p_i). 
    # với y_i là nhãn đúng (one-hot), nên chỉ có một y_c = 1 (lớp đúng), còn các y_i khác = 0.
    # thì ví dụ phân biệt mèo gà chó ở trên: với nhãn đúng là mèo (lớp 0), y = [1, 0, 0], p = [0.645, 0.322, 0.032]
    # H(y, p) = -[y_0 * log(p_0) + y_1 * log(p_1) + y_2 * log(p_2)] 
    # = -[1 * log(0.645) + 0 * log(0.322) + 0 * log(0.032)] = -log(0.645). 
    # Các nhãn khác bị triệt tiêu vì nhân với 0.
    return (-np.log(probs)).tolist()


def num_grad(f, x: list[float], h: float = 1e-5) -> list[float]:
    """Yêu cầu 5 — đạo hàm số theo TỪNG chiều bằng central difference.

    g[i] = (f(x + h*e_i) - f(x - h*e_i)) / (2h)   với e_i là vector đơn vị chiều i.
    Sai số O(h^2): chuỗi Taylor thấy số hạng h^2 triệt tiêu khi trừ hai vế.
    """
    x_arr = np.array(x, dtype=float)
    g = np.zeros_like(x_arr)
    for i in range(len(x_arr)):
        # QUY TẮC VÀNG: nhích TỪNG chiều một. Copy x, chỉ cộng/trừ h vào chiều i.
        x_cong = x_arr.copy()
        x_tru = x_arr.copy()
        x_cong[i] += h
        x_tru[i] -= h
        g[i] = (f(x_cong) - f(x_tru)) / (2 * h)
    return g.tolist()


def descent(f, x0: list[float], lr: float = 0.2, steps: int = 10) -> list[list[float]]:
    """Yêu cầu 6 — vòng gradient descent dùng chính num_grad.

    x_{k+1} = x_k - lr * grad(x_k)  (đi NGƯỢC hướng tăng nhanh nhất để giảm f).
    Trả về toàn bộ quỹ đạo (trace) để in từng bước.
    """
    x = np.array(x0, dtype=float)
    trace = [x.copy().tolist()]
    for _ in range(steps):
        grad = np.array(num_grad(f, x.tolist()))
        x = x - lr * grad
        trace.append(x.copy().tolist())
    return trace


def main():
    vec1 = [0.9, 0.8]
    vec2 = [0.8, -0.9]
    similarity = cosine(vec1, vec2)
    print(f"Cosine similarity between {vec1} and {vec2} is: {similarity}")
    print("=" * 50)
    
    # ví dụ về softmax (logits lớn để thấy tính ổn định số)
    logits = [20, 10, 3.0]
    probabilities = softmax(logits, T=2)
    print(f"Softmax probabilities for {logits} are: {probabilities}")
    print(f"With very large logits {[1000, 1000, 999]} (no NaN thanks to max-subtraction): {softmax([1000, 1000, 999])}")

    # requirement 3 — Temperature experiment with logits [2, 1, 0.1]
    exp_logits = [2, 1, 0.1]
    print(f"\n--- Temperature experiment with logits {exp_logits} ---")
    for T in [0.5, 1.0, 2.0]:
        print(f"T = {T}: {softmax(exp_logits, T=T)}")
    print("Explanation: T divides the logits BEFORE the exp — a small T amplifies the gaps")
    print("between logits, so the probability mass concentrates on the largest logit (sharp/")
    print("one-hot peak); a large T shrinks those gaps, flattening the distribution toward uniform.")

    # requirement 4 — cross-entropy table check: -log(p)
    p_true = [0.9, 0.5, 0.01]
    losses = cross_entropy(p_true)
    expected = [0.105, 0.693, 4.605]
    print(f"\n--- Cross-entropy table check: -log(p) ---")
    for p, loss in zip(p_true, losses):
        print(f"-log({p}) = {loss:.3f}  (expected 0.105 / 0.693 / 4.605)")
    # bảng lưu số làm tròn (ln(10/9)=0.10536, ln2=0.6931, ln100=4.6052) --> cần atol
    np.testing.assert_allclose(losses, expected, atol=0.001)
    print("Assert -log(p) matches theoretical table: PASSED")

    # --- requiremente 5-7: num_grad + gradient descent ---
    f_sq = lambda v: v[0] ** 2 + v[1] ** 2  # f(x, y) = x^2 + y^2
    x0 = [3.0, 4.0]

    # requirement 5 — cross-check num_grad vs analytical gradient (2x, 2y) = (6, 8)
    grad_num = num_grad(f_sq, x0)
    print(f"\nnum_grad on f(x,y)=x^2+y^2 at (3,4) = {grad_num}  (analytical = [6.0, 8.0])")
    np.testing.assert_allclose(grad_num, [6.0, 8.0], atol=1e-6)
    print("Assert num_grad ~ analytical gradient: PASSED")

    # requirement 6 — trace gradient descent from (3,4), lr=0.2, 10 steps
    trace = descent(f_sq, x0, lr=0.2, steps=10)
    print(f"\n--- Gradient descent on f(x,y)=x^2+y^2 from {x0}, lr=0.2 ---")
    for k, pos in enumerate(trace):
        print(f"step {k}: pos = ({pos[0]:+.4f}, {pos[1]:+.4f}), f = {f_sq(pos):.6f}")
    print("-> positions shrink toward (0, 0)")

    # requirement 7 — learning rate experiments: 'snail' (too small) vs 'fly off the hill' (too big)
    for lr, note in [
        (0.01, "snail's pace: factor (1-2*0.01)=0.98 per step, barely crawls toward origin"),
        (1.1, "flies off: factor (1-2*1.1)=-1.2, |x| grows x1.2 with sign flip every step"),
    ]:
        trace = descent(f_sq, x0, lr=lr, steps=10)
        print(f"\n--- lr = {lr} ---")
        for k, pos in enumerate(trace):
            print(f"step {k}: pos = ({pos[0]:+.4f}, {pos[1]:+.4f}), f = {f_sq(pos):.6f}")
        print(f"Observation: {note}")

if __name__ == "__main__":
    main()
    