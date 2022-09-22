OUTDIR=media/videos

${OUTDIR}/scene01_title/1080p60/Title.mp4: scene01_title.py
	manim render scene01_title.py Title -q h

${OUTDIR}/scene02_scenarios/1080p60/Scenarios.mp4: scene02_scenarios.py
	manim render scene02_scenarios.py Scenarios -q h

.PHONY: all_hq

all_hq: ${OUTDIR}/scene01_title/1080p60/Title.mp4 \
		${OUTDIR}/scene02_scenarios/1080p60/Scenarios.mp4
