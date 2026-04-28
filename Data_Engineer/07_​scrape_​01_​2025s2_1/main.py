from bs4 import BeautifulSoup
def Q1(file_path="2566.html"): # DO NOT modify this line
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "lxml")
    divs = soup.select("div.bud-day-col")
    d = {
        "วันจันทร์ที่": 0,
        "วันอังคารที่": 1,
        "วันพุธที่": 2,
        "วันพฤหัสบดีที่": 3,
        "วันศุกร์ที่": 4,
        "วันเสาร์ที่": 5,
        "วันอาทิตย์ที่": 6,
    }
    results = [0 for i in range(7)]
    for div in divs:
        if "ที่" in div.text.strip():
            dw = div.text.strip().split()[0]
            results[d[dw]]+=1
    return results
def Q2(file_path='2566.html'): # DO NOT modify this line
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "lxml")
    divs = soup.select("div.bud-day")
    for div in divs:
        if "วันวิสาขบูชา" in div.text.strip():
            child = list(div.children)
            return child[0].text.strip()
    return None
print(Q1()) # do not delete this line
print(Q2())