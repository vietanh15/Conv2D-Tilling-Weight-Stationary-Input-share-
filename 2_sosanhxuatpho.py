import matplotlib.pyplot as plt

def compare_files_larger_points(file1, file2):
    print("Đang đọc và xử lý dữ liệu...")
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file! Vui lòng kiểm tra lại đường dẫn: {file1} hoặc {file2}")
        return

    max_len = max(len(lines1), len(lines2))
    diff_indices = [] 
    diff_values = [] 

    for i in range(max_len):
        str1 = lines1[i].strip() if i < len(lines1) else None
        str2 = lines2[i].strip() if i < len(lines2) else None
        try:
            # Chuyển đổi từ số thập phân (thay vì hex)
            val1 = int(str1) if str1 else 0
            val2 = int(str2) if str2 else 0
            diff = abs(val1 - val2)
            if diff > 0:
                diff_indices.append(i + 1)
                diff_values.append(diff)
        except (ValueError, TypeError):
            # Bỏ qua các dòng không hợp lệ
            print(f"Cảnh báo: Không thể xử lý dòng {i+1}. Bỏ qua.")
            pass

    total_diff = len(diff_values)
    print(f"Xử lý xong. Tổng số lỗi: {total_diff}")

    if total_diff > 0:
        plt.figure(figsize=(14, 6), dpi=100)
        
        plt.scatter(diff_indices, diff_values, 
                    s=15,
                    c='tab:blue',
                    marker='o',
                    alpha=0.3,
                    label='Điểm sai lệch')

        plt.title(f'Phổ độ lệch giữa hai file\nTổng số mẫu sai khác: {total_diff:,} mẫu')
        plt.xlabel('Vị trí dòng (Line Index)')
        plt.ylabel('Độ lệch tuyệt đối (Absolute Error)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if diff_values:
            plt.ylim(0, max(diff_values) * 1.1)

        plt.tight_layout()
        print("Đang hiển thị biểu đồ...")
        plt.show()
    else:
        print("Hai file giống hệt nhau!")

# ==== CHẠY THỬ ====
# So sánh file output từ input sharing với file golden
compare_files_larger_points("ofm_weight_stationary.txt", "ofm_golden.txt")