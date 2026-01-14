import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.subplot(1, 1, 1)
plt.subplots_adjust(bottom=0.2, left=0.15)
fs = 20
lw = 3
mk = 'o'
ms = 15
region_cnt = [3, 4, 5, 6, 7]
bedges_cnt = [26, 16, 74, 56, 142]
action_cnt = [bedges_cnt[i] / region_cnt[i] for i in range(5)]

plt.plot(region_cnt, action_cnt, linewidth=lw, marker=mk, markersize=ms)
plt.xlabel('Region Count', fontsize=fs)
plt.ylabel('Average Action Space', fontsize=fs)
plt.xticks(region_cnt, region_cnt, fontsize=fs)
plt.yticks(fontsize=fs)
plt.show()
