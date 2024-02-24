from matplotlib import pyplot as plt

dice_coefficient = [0.8206, 0.8094, 0.8115, 0.8142]
fcm_flavour = ['fcm', 'fgfcm', 'enfcm', 'fcm_s1']
plt.ylim(0.80, 0.83)
plt.bar(fcm_flavour, dice_coefficient)
plt.xlabel('FCM Method')
plt.ylabel('Dice Coefficients')
plt.title('Dice Coefficient vs Method')
plt.show()