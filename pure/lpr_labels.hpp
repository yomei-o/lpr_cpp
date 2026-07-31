// Label tables + decode for the 9 LPR heads (from Classificator/lpr/lpr.py). Turns the 9 argmax
// indices into the plate string: 地域名 分類番号 ひらがな 一連番号.
#pragma once
#include <string>
#include <vector>
#include <array>

namespace lpr {
inline const std::vector<std::string> REGION = {
  "札幌","函館","旭川","室蘭","苫小牧","釧路","知床","帯広","北見","青森","弘前","八戸","岩手","盛岡","平泉","宮城","仙台","秋田","山形","庄内","福島","会津","郡山","白河","いわき","水戸","土浦","つくば","宇都宮","那須","とちぎ","群馬","前橋","高崎","大宮","川口","所沢","川越","熊谷","春日部","越谷","千葉","成田","習志野","市川","船橋","袖ヶ浦","市原","野田","柏","松戸","品川","世田谷","練馬","杉並","板橋","足立","江東","葛飾","八王子","多摩","横浜","川崎","湘南","相模","山梨","富士山","新潟","長岡","上越","長野","松本","諏訪","富山","石川","金沢","福井","岐阜","飛騨","静岡","浜松","沼津","伊豆","名古屋","豊橋","三河","岡崎","豊田","尾張小牧","一宮","春日井","三重","鈴鹿","四日市","伊勢志摩","滋賀","京都","なにわ","大阪","和泉","堺","奈良","飛鳥","和歌山","神戸","姫路","鳥取","島根","出雲","岡山","倉敷","広島","福山","山口","下関","徳島","香川","高松","愛媛","高知","福岡","北九州","久留米","筑豊","佐賀","長崎","佐世保","熊本","大分","宮崎","鹿児島","奄美","沖縄"};
// hiragana list (53) — multi-byte chars, kept as UTF-8 strings
inline const std::vector<std::string> HIRA = {
  "あ","い","う","え","か","き","く","け","こ","さ","す","せ","そ","た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","ほ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","A","B","C","E","H","K","L","M","T","Y","V"};
inline const std::string CN01 = "0123456789";
inline const std::string CN02 = "0123456789ACFHKLMPXY";
inline const std::string CN03 = "0123456789ACFHKLMPXY ";

// heads order: [region, class_num_01, class_num_02, class_num_03, hiragana, plate_num_01..04]
struct Plate { std::string region, cls, hira, num; };
inline Plate decode(const std::array<int, 9>& a) {
  Plate p;
  p.region = (a[0] >= 0 && a[0] < (int)REGION.size()) ? REGION[a[0]] : "?";
  if (a[1] < (int)CN01.size()) p.cls += CN01[a[1]];
  if (a[2] < (int)CN02.size()) p.cls += CN02[a[2]];
  if (a[3] < (int)CN03.size() && CN03[a[3]] != ' ') p.cls += CN03[a[3]];
  p.hira = (a[4] >= 0 && a[4] < (int)HIRA.size()) ? HIRA[a[4]] : "?";
  for (int i = 5; i < 9; ++i) if (a[i] != 10) p.num += std::to_string(a[i]);
  return p;
}
}  // namespace lpr
