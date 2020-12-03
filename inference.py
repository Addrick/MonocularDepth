import matplotlib.pyplot as plt
# black and white
bw_img = img.reshape(1,75,284,1)
pred = model.predict(bw_img)
pred *= 255/pred.max()
pred = pred.astype(np.uint8)
pred = pred.reshape(75,284)
plt.imshow(pred, 'gray')
plt.show()