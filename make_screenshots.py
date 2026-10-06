"""Снимки экрана интерфейса для п. 3.2 (нужен запущенный сервер Streamlit на порту 8501).

Запуск: .venv\\Scripts\\python make_screenshots.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501/"
OUT = Path(__file__).parent / "docs"
OUT.mkdir(exist_ok=True)


def wait_ready(page, text):
    page.get_by_text(text, exact=False).first.wait_for(timeout=60000)
    page.wait_for_timeout(2500)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge")
    ctx = browser.new_context(viewport={"width": 1400, "height": 1000}, device_scale_factor=2, color_scheme="light")
    page = ctx.new_page()

    # рис. 2: ввод данных пациента
    page.goto(URL)
    wait_ready(page, "Данные пациента")
    page.get_by_text("очень сильно", exact=True).first.wait_for(timeout=60000)
    page.wait_for_timeout(3000)
    page.screenshot(path=OUT / "fig2_input.png")

    # рис. 3: результат с раскрытым пояснением
    page.get_by_text("Почему такой ответ", exact=False).first.click()
    page.wait_for_timeout(1500)
    res = page.get_by_text("Результат", exact=True).first
    res.scroll_into_view_if_needed()
    page.wait_for_timeout(1000)
    main = page.locator("section.stMain, section[data-testid='stMain']").first
    box = res.bounding_box()
    page.screenshot(path=OUT / "fig3_result.png", full_page=True,
                    clip={"x": 300, "y": box["y"] + page.evaluate("window.scrollY") - 20, "width": 1100, "height": 720})

    # рис. 4: эталоны
    page.goto(URL + "?page=Эталоны")
    wait_ready(page, "Сходство эталонов между собой")
    page.screenshot(path=OUT / "fig4_etalons.png", full_page=True)

    # рис. 5: оценка качества на синтетической выборке
    page.goto(URL + "?page=Оценка качества")
    wait_ready(page, "Источник данных")
    page.get_by_text("Синтетическая выборка", exact=False).first.click()
    wait_ready(page, "Матрица ошибок")
    page.wait_for_timeout(4000)
    page.screenshot(path=OUT / "fig5_quality.png", full_page=True)
    browser.close()
print("готово:", sorted(x.name for x in OUT.glob("fig*.png")))
