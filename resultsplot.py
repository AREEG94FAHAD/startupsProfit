import matplotlib.pyplot as plt

fig, ax = plt.subplots()


def plotResults(xvalues, yvalues, title, ylabel):
    bar_labels = ['red', 'blue', 'orange']
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    ax.bar(xvalues, yvalues, label=bar_labels)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend(title)
    plt.grid()
    plt.show()

x = ['Two x', 'Three x', 'Four x']
yR2 = [94.69, 93.94, 93.47]

titleR2 = "R2_score Results"
y_R2 = 'R2_Score'

# plotResults(x, yR2, titleR2, y_R2)


plotResults(x, [0.008621999999999908,0.009375899999999993,0.011160599999999965],'Consumption Delay', ' Consumption delay in sec' )

# delay results
# 0.011160599999999965 x4
# 0.009375899999999993 x3
# 0.008621999999999908 x2