vismatrix  
Tool dùng visualize kết quả dành cho classification của AIC Hackathon.  
  
I. Chuẩn bị dữ liệu:  
- Yêu cầu hai đường dẫn là ```imgpath``` và ```labelpath```
  
II. Cách chạy:  
    $apt install python3 python3-pip  
    $pip3 install flask flask-restful flask-cors opencv-python  
    $python3 web.py  
    Mở trình duyệt web vào link và sử dụng.  
III. Giải thích URL:  
    URL tool dạng http://localhost:8000/thumbnailimg?index=0&imgpath=&labelpath=  
    Với:  
        +index=0 là vị trí của trang hiện tại  
        +imgpath là đường dẫn tới folder ảnh  
        +labelpath là đường dẫn tới folder chứa file ```visual.txt```   