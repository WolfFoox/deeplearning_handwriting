import cv2
import os

# 指定输入文件夹和输出文件夹
input_folder = "RAW"
output_folder = "COOKED"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历指定文件夹中的图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):

        # 读取图像
        image = cv2.imread(os.path.join(input_folder, filename))

        # 增强对比度和亮度
        alpha = 1.1  # 增强亮度
        beta = 30    # 增强对比度
        enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # 如果图像是灰度图像，将其转换为彩色图像
        if len(enhanced_image.shape) == 2:
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

        # 保存处理后的图像到输出文件夹
        output_filename = os.path.join(output_folder, filename)
        cv2.imwrite(output_filename, enhanced_image)

        print(f'Processed: {filename}')

print('图像处理完成')
