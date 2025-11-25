import pandas as pd
import os

def create_dataframe(sdir):
    classlist = sorted(os.listdir(sdir))
    filepaths = []
    labels = []
    for klass in classlist:
        classpath = os.path.join(sdir, klass)
        flist = sorted(os.listdir(classpath))
        for f in flist:
            fpath = os.path.join(classpath, f)
            filepaths.append(fpath)
            labels.append(klass)
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df


# os.listdir(path)
# Trả về danh sách tên file/thu mục (không kèm đường dẫn) trong thư mục path.

# sorted(iterable)
# Sắp xếp iterable (ở đây dùng để có thứ tự cố định cho class list và file list).

# os.path.join(a, b, ...)
# Nối các thành phần đường dẫn an toàn theo hệ điều hành (ví dụ "/content" + "AID" → "/content/AID").

# list.append(x)
# Phương thức của list Python — thêm phần tử x vào cuối list (mutates list).

# pd.Series(data, name='...')
# Tạo một Series pandas từ data (mảng/list). Tham số name gán tên cột khi convert thành DataFrame.

# pd.concat([ser1, ser2], axis=1)
# Nối (concatenate) các Series/DataFrame theo chiều cột (axis=1) để tạo DataFrame với 2 cột (filepaths, labels).

# Kết quả hàm create_dataframe(sdir) là một pandas.DataFrame hai cột:

# filepaths: đường dẫn đầy đủ tới mỗi ảnh
# labels: tên thư mục (class) tương ứng
# Gợi ý nhanh: nên lọc file theo đuôi ảnh trong vòng lặp (ví dụ .jpg/.png) để tránh đọc file không phải ảnh.