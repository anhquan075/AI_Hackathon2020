vismatrix  
Tool dùng visualize kết quả dành cho detection.  
  
I. Chuẩn bị dữ liệu:  
    -Dữ liệu được chia thành 2 folder chứa ảnh và label.  
    -Ảnh lưu dưới dạng .jpg và label là .txt, tên file ảnh và file label phải tương ứng nhau.  
    -Cấu trúc của mỗi file label: gồm nhiều dòng, mỗi dòng là một bbox, trên mỗi dòng là x1,y1,x2,y2,cls  
    -Trong source kèm file cvt_l1.py hỗ trợ chuyển định dạng từ yolo sang dạng của tool.  
  
II. Cách chạy:  
    $apt install python3 python3-pip  
    $pip3 install flask flask-restful flask-cors opencv-python  
    $python3 web.py  
    Mở trình duyệt web vào link và sử dụng.  
III. Giải thích URL:  
    URL tool dạng http://192.168.24.45/thumbnailimg?index=0&imgpath=&labelpath=  
    Với:  
        +index=0 là vị trí của trang hiện tại  
        +imgpath là đường dẫn tới folder ảnh  
        +labelpath là đường dẫn tới folder label  
      
         
IV. Thêm phần visualize classification của AIC Hackathon
    
  
  
  
  
  
  

