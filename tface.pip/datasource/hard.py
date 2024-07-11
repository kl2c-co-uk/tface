
target_width = 1920
target_height = 1080
heatmap_scale = 0.1

def sizes():
	return (
		target_width,
		target_height,
		int(heatmap_scale * target_width),
		int(heatmap_scale * target_height)
	)


