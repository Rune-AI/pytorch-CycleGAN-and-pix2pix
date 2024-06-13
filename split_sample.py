from PIL import Image



filename  = "D:\DAYIII\MLErosionCustom\\training_images\\test\\1175.png"

AB = Image.open(filename)

w, h = AB.size
w2 = int(w / 2)
A = AB.crop((0, 0, w2, h))
B = AB.crop((w2, 0, w, h))

# save a and b

A.save(".\\results\\A.png")
B.save(".\\results\\B.png")