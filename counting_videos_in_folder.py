import os
videos_folder = 'D:\project\\frames'

index = 1

for filename in os.listdir(videos_folder):
    index += 1


print(index)