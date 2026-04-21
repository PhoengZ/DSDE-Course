from bs4 import BeautifulSoup

def Q1(file_path):
    with open(file_path, 'r') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'lxml')
    bud_days = soup.select("div.bud-day")
    d = {
        "วันจันทร์ที่":0,"วันอังคารที่":0, "วันพุธที่":0, "วันพฤหัสบดีที่":0, "วันศุกร์ที่":0, "วันเสาร์ที่":0, "วันอาทิตย์ที่":0
    }
    for div in bud_days:
        e = div.select_one("div.bud-day-col:-soup-contains('ที่')")
        day, _, _, _ = e.text.strip().split()
        d[day]+=1
    return list(d.values())

def Q2(file_path):
    with open(file_path, 'r') as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    divs = soup.select('div.bud-day-col:-soup-contains("วัน")')
    for i, div in enumerate(divs):
        if "วันวิสาขบูชา" in div.text.strip():
            return divs[i-1].text.strip()
    return None

exec(input().strip())