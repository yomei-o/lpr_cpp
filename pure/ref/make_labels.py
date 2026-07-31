# Convert human plate annotations to the labels.txt index format that lpr_data.hpp / train_lpr expect.
# Reads a TSV (tab-separated) with columns:  filename  region  class_num  hiragana  plate_num
#   e.g.   plate0001.jpg <TAB> 品川 <TAB> 500 <TAB> あ <TAB> 1234
# Writes labels.txt lines:  <filename> <9 indices>
#   (region, class_num_01/02/03, hiragana, plate_num_01/02/03/04)
# Blanks: class_num shorter than 3 -> pad with ' ' (index in class_num_03); plate_num shorter than 4
# -> right-aligned, missing high digits -> 10 (blank). Unknown chars -> the head's last (blank) class.
#   python make_labels.py annotations.tsv > dataset/labels.txt
import sys

REGION = ['札幌','函館','旭川','室蘭','苫小牧','釧路','知床','帯広','北見','青森','弘前','八戸','岩手','盛岡','平泉','宮城','仙台','秋田','山形','庄内','福島','会津','郡山','白河','いわき','水戸','土浦','つくば','宇都宮','那須','とちぎ','群馬','前橋','高崎','大宮','川口','所沢','川越','熊谷','春日部','越谷','千葉','成田','習志野','市川','船橋','袖ヶ浦','市原','野田','柏','松戸','品川','世田谷','練馬','杉並','板橋','足立','江東','葛飾','八王子','多摩','横浜','川崎','湘南','相模','山梨','富士山','新潟','長岡','上越','長野','松本','諏訪','富山','石川','金沢','福井','岐阜','飛騨','静岡','浜松','沼津','伊豆','名古屋','豊橋','三河','岡崎','豊田','尾張小牧','一宮','春日井','三重','鈴鹿','四日市','伊勢志摩','滋賀','京都','なにわ','大阪','和泉','堺','奈良','飛鳥','和歌山','神戸','姫路','鳥取','島根','出雲','岡山','倉敷','広島','福山','山口','下関','徳島','香川','高松','愛媛','高知','福岡','北九州','久留米','筑豊','佐賀','長崎','佐世保','熊本','大分','宮崎','鹿児島','奄美','沖縄']
HIRA = list('あいうえかきくけこさすそせたちつてとなにぬねのはひふほまみむめもやゆよらりるれろわをABCEHKLMTYV')
CN01 = list('0123456789'); CN02 = list('0123456789ACFHKLMPXY'); CN03 = list('0123456789ACFHKLMPXY ')

def region_id(s): return REGION.index(s) if s in REGION else 0
def hira_id(s):   return HIRA.index(s) if s in HIRA else 0
def class_ids(s):
    s = (s or '').ljust(3)                                   # left-justify, pad with ' '
    def idx(lst, ch): return lst.index(ch) if ch in lst else len(lst) - 1
    return [idx(CN01, s[0]), idx(CN02, s[1]), idx(CN03, s[2] if s[2] != ' ' else ' ')]
def plate_ids(s):
    s = ''.join(ch for ch in (s or '') if ch.isdigit())[-4:].rjust(4, ' ')   # right-align to 4
    return [10 if ch == ' ' else int(ch) for ch in s[:3]] + [int(s[3]) if s[3].isdigit() else 0]

for line in sys.stdin if len(sys.argv) < 2 else open(sys.argv[1], encoding='utf-8'):
    line = line.rstrip('\n')
    if not line or line.startswith('#'): continue
    parts = line.split('\t')
    if len(parts) < 5: continue
    fn, region, cn, hira, pn = parts[:5]
    ids = [region_id(region)] + class_ids(cn) + [hira_id(hira)] + plate_ids(pn)
    # order in labels.txt = region, class_num_01/02/03, hiragana, plate_num_01..04
    ids = [ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8]]
    print(fn, *ids)
