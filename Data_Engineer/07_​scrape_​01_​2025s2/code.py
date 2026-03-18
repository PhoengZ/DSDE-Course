from bs4 import BeautifulSoup

def Q1(file_path):
    with open(file_path, "r")as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    divs = soup.select("div.bud-day-col")
    check = ["วันจันทร์", "วันอังคาร", "วันพุธ", "วันพฤหัสบดี", "วันศุกร์", "วันเสาร์", "วันอาทิตย์"]
    result = [0 for i in range(7)]
    for day in divs:
        for i, c in enumerate(check):
            if c in day.text.strip():
                result[i]+=1
    return result

def Q2(file_path):
    with open(file_path, "r") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    all_div = soup.select("div.bud-day-col")
    for i, e in enumerate(all_div):
        a = e.find("a")
        if a and "วันวิสาขบูชา" in a.text.strip():
            return all_div[i-2].text.strip()
    return None

exec(input().strip())