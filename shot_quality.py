from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright
OUT = Path(__file__).parent / "docs"
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    pg = b.new_context(viewport={"width": 1400, "height": 2800}, device_scale_factor=2, color_scheme="light").new_page()
    pg.goto("http://localhost:8501/?page=Оценка качества")
    pg.get_by_text("Источник данных").first.wait_for(timeout=60000)
    pg.wait_for_timeout(2000)
    pg.get_by_text("Синтетическая выборка", exact=False).first.click()
    pg.get_by_text("Матрица ошибок", exact=False).first.wait_for(timeout=120000)
    pg.wait_for_timeout(6000)
    r1 = pg.get_by_text("Случаев:", exact=False).first.bounding_box()
    r2 = pg.get_by_text("Сравнение с логистической регрессией", exact=False).first.bounding_box()
    tmp = OUT / "_full.png"
    pg.screenshot(path=tmp)
    b.close()
img = Image.open(tmp)
y0, y1 = int((r1["y"] - 10) * 2), int((r2["y"] + 330) * 2)
x0, x1 = int((r1["x"] - 10) * 2), img.width - 2 * 60
img.crop((x0, y0, x1, min(y1, img.height))).save(OUT / "fig5_quality.png")
tmp.unlink()
print("ok", x0, y0, x1, y1)
