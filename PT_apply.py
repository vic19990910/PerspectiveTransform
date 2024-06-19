import cv2
import numpy as np

# 定義全局變數來存儲滑鼠點擊的座標
points = []

# 定義滑鼠事件的回調函數
def click_event(event, x, y, flags, param):
    global points
    # 如果按下左鍵，則將點的座標存儲到列表中
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # 在圖像上繪製點
        cv2.circle(phototest, (x, y), 30, (0, 0, 255), -1)
        cv2.imshow("Image", phototest)
        # 如果收集到四個點，則執行透視轉換
        if len(points) == 4:
            apply_perspective_transformation()

# 定義透視轉換函數
def apply_perspective_transformation():
    global points
    # 定義四個點的座標
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640,480]])  # 目標影像的四個角點

    # 計算透視轉換矩陣
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 執行透視轉換
    transformed_frame = cv2.warpPerspective(phototest, matrix, (640, 480))
     # 設定保存文件名
    saved_file = f"{filename}_transformed.jpg"
    # 寫入操作
    success = cv2.imwrite(saved_file, transformed_frame)

    # 檢查返回值
    if success:
        print("文件保存成功！")
    else:
        print("文件保存失敗！")

    # 顯示轉換後的影像
    cv2.imshow("Transformed Image", transformed_frame)

    cv2.waitKey(0)

# 讀取影像
filename = input("enter the photo name :  ")
phototest = cv2.imread(f"{filename}.jpg")
phototest = cv2.resize(phototest, (8688,5792))
# 建立可以調整大小的視窗
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

# 顯示影像並設置滑鼠事件的回調函數
cv2.imshow("Image", phototest)
cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
