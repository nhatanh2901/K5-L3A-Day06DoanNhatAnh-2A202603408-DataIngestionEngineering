# Báo cáo cá nhân — Day 6

**Sinh viên:** Đoàn Nhật Anh — **MSSV:** 2A202603408
**Ngày chạy gần nhất:** 13/11/2026 (giờ Việt Nam)
**Trạng thái:** Đã chạy đủ ba cấu hình trên tập dev và test; báo cáo có kết quả retrieval, kiểm tra metadata và đánh giá hỗ trợ AI cho coherence/grounding.

## 1. Mục tiêu và phạm vi

Xây dựng pipeline ingest PDF, chia chunk, tạo embedding và tìm kiếm top-3; so sánh ảnh hưởng của cách chia chunk (A–B) và metadata filter (B–C) trong điều kiện giữ nguyên corpus, model, vector store và benchmark. Các chỉ số retrieval được tính riêng cho dev và test; test không được dùng để chọn chiến lược.

## 2. Corpus và xử lý dữ liệu

- **Chủ đề/tài liệu:** *Phân tích hành vi mua sắm Instacart trên Big Data*, file `starter/data/Report Data.pdf`.
- **Quy mô thực đo:** 20 trang vật lý; 19 trang được tính là nội dung hữu ích. Trang được tính khi có ít nhất 150 ký tự trích xuất; trang tài liệu tham khảo bị loại. Không nhân bản trang.
- **Kết luận provenance/quyền:** Người học xác nhận có quyền chia sẻ PDF và các trích đoạn trên repository GitHub công khai. Bản PDF không ghi URL nguồn hoặc tên giấy phép cụ thể; [manifest](../../data/manifest.json) lưu xác nhận quyền chia sẻ này, không tuyên bố PDF có giấy phép mở.
- **Lưu ý về ngưỡng trang:** GUIDE/RUBRIC trong repository ghi mốc 30 trang nội dung. Theo thông tin người học cung cấp sau khi trao đổi với giảng viên, mốc này không bắt buộc cho bài nộp hiện tại. Báo cáo vẫn ghi đúng corpus thực tế là 19 trang và không tuyên bố đã đạt mốc 30.
- **Trích xuất và chuẩn hóa:** Dùng pypdf, trích riêng từng trang, chuẩn hóa Unicode NFC và khoảng trắng; giữ page, section, source và offset để truy xuất lại nội dung. Trang không đủ nội dung không đưa vào tập retrieval.
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, chạy local, 384 chiều, giới hạn 128 token. Cùng model được dùng cho document và query.
- **Vector store:** Chroma PersistentClient 1.5.9, metric cosine, dữ liệu persistent tại `results/chroma/`. Python 3.13.5, pypdf 6.21.4, sentence-transformers 5.7.0. Embedding không gọi dịch vụ trả phí nên chi phí dịch vụ ghi nhận là 0; thời gian là số đo của lần chạy local và có thể thay đổi theo máy/cache.
- **Metadata chunk:** `document_id`, `source`, `page`, `section`, `start_offset`, `end_offset`. `section` được suy ra từ heading trang PDF; dùng làm filter khi câu hỏi nêu rõ phạm vi chương.

## 3. Thiết kế benchmark và cấu hình

Bộ [benchmark](../../benchmark/questions.json) có 20 câu đã định nghĩa trước khi đánh giá: 15 dev và 5 test; có 4 câu dùng metadata filter, trong đó có 1 câu test. Evidence được kiểm tra tự động có mặt trên trang nguồn gold. Câu test `q17` yêu cầu giới hạn kết quả trong Chương 4.

| Cấu hình | Chunking | Retrieval |
|---|---|---|
| A | Cửa sổ cố định 300 ký tự, overlap 40 | Dense top-3, không filter |
| B | Gom dòng/đoạn theo nguồn, tối đa 300 ký tự | Dense top-3, không filter |
| C | Dùng nguyên chunk và embedding của B | Dense top-3; filter `section` khi câu hỏi cung cấp |

B–C chỉ thay đổi việc áp dụng filter. Quy tắc chọn được chốt theo dev: Evidence Hit@3 cao nhất, sau đó Hit@1 cao hơn, rồi ít chunk hơn. Dev chọn A (8/15 Evidence Hit@3); lựa chọn này được ghi trước lượt test gần nhất.

## 4. Kết quả retrieval

Hit@1 và Evidence Hit@3 được báo theo số câu đúng trên tổng số câu của split. Ingest time gồm các bước ingest được ghi trong stats, không gồm thời gian người dùng chờ tải model.

| Strategy | Dev Hit@1 | Dev Evidence Hit@3 | Chunks | Dev ingest | Test Hit@1 | Test Evidence Hit@3 | Test ingest | Chi phí dịch vụ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2/15 (13,3%) | 8/15 (53,3%) | 142 | 3,535 s | 2/5 (40%) | 3/5 (60%) | 3,630 s | 0 |
| B | 3/15 (20%) | 6/15 (40%) | 138 | 3,319 s | 3/5 (60%) | 3/5 (60%) | 3,294 s | 0 |
| C | 3/15 (20%) | 6/15 (40%) | 138 | 3,282 s | 3/5 (60%) | 3/5 (60%) | 3,264 s | 0 |

File kết quả chi tiết: [dev metrics](../../results/dev_metrics.csv), [test metrics](../../results/test_metrics.csv), [dev traces](../../results/dev_traces.jsonl), [test traces](../../results/test_traces.jsonl), [dev stats](../../results/dev_ingestion_stats.json), [test stats](../../results/test_ingestion_stats.json).

## 5. Phân tích năm khía cạnh

### 5.1 Retrieval accuracy

Trên dev, A dẫn đầu Evidence Hit@3 với 8/15 câu, trong khi B và C đạt 6/15. Trên test, cả ba cùng đạt 3/5 Evidence Hit@3; Hit@1 của B/C là 3/5, cao hơn A (2/5). Vì test có năm câu, mỗi câu làm thay đổi tỷ lệ 20 điểm phần trăm; kết luận được giới hạn ở bộ câu hỏi và tài liệu đã chạy.

Hai lỗi nổi bật, đều xảy ra ở cả ba cấu hình:

- **q16:** đáp án gold là “bốn đặc trưng số”, trang 15; top-3 không tìm thấy evidence gold.
- **q19:** câu hỏi về cách giảm rủi ro job pair co-occurrence quá nặng, trang 18; top-3 không tìm thấy evidence gold.

Trace top-3 để kiểm tra các lỗi này nằm trong `test_traces.jsonl`.

### 5.2 Chunk coherence

[Coherence audit](../../results/coherence_audit.csv) chấm cùng 10 vị trí nguồn cho mỗi cấu hình theo thang 0–2. Kết quả đánh giá hỗ trợ AI: A **0,4/2**, B **1,3/2**, C **1,3/2**. A bị trừ điểm chủ yếu do cửa sổ cố định cắt giữa từ/câu; B/C thường giữ được ranh giới dòng hoặc câu tốt hơn. B và C có cùng chunk nên điểm bằng nhau. Điểm và lý do từng mẫu được ghi trong CSV, gắn nhãn AI-assisted; đây không phải đánh giá độc lập của người chấm.

### 5.3 Metadata usefulness

Trace lưu source, page, section và điểm truy xuất. Đối chiếu 45 kết quả top-3 với trang PDF cho thấy `document_id`, `source`, `page` hợp lệ và `section` khớp trang nguồn ở **45/45** lượt. [Metadata audit](../../results/metadata_audit.csv) lưu chi tiết từng chunk.

Với `q17`, có 3/3 kết quả của A, 2/3 của B và 3/3 của C thuộc đúng section Chương 4. Theo tiêu chí nghiêm ngặt là cả top-3 đều phải tuân thủ, số câu đạt là A: **1/1**, B: **0/1**, C: **1/1**. A đạt ở trường hợp này dù không áp filter; C đạt do filter được thực thi. Kết quả mô tả này xác định hành vi của ba cấu hình trên q17; số câu filter test là một nên không đại diện cho mọi truy vấn.

### 5.4 Grounding quality

Evidence Hit@3 là phép đo retrieval tự động, không phải độ chính xác câu trả lời của LLM. [Grounding audit](../../results/grounding_audit.csv) đánh giá mức evidence theo thang 0–2 với lượt rà soát hỗ trợ AI: cả A/B/C đều có **1 câu ở mức 0, 1 câu ở mức 1, 3 câu ở mức 2; trung bình 1,4/2**. `q16` không có bằng chứng trả lời (0); `q19` có khuyến nghị liên quan nhưng thiếu nội dung cần thiết (1); `q17`, `q18`, `q20` có evidence đủ trả lời (2). Từng điểm kèm lý do trong CSV; đây là đánh giá hỗ trợ AI, không phải nhãn do người chấm độc lập xác nhận.

### 5.5 Data strategy impact và trade-off

- **A so với B:** A tạo 142 chunk, nhiều hơn B 4 chunk, nhưng có Evidence Hit@3 dev cao hơn 2 câu. Trên test, A có Hit@1 thấp hơn B một câu; Evidence Hit@3 bằng nhau.
- **B so với C:** cùng 138 chunk và cùng chunking; filter không đổi tổng Hit@1/Evidence Hit@3 của split. Ở câu filter test `q17`, C tuân thủ filter, còn B không tuân thủ.
- Theo quy tắc đã đặt trước, chọn A dựa trên Evidence Hit@3 dev. Kết quả test không được dùng để đổi lựa chọn. Giới hạn chính là test chỉ có 5 câu, corpus có một nguồn, và điểm coherence/grounding là rà soát hỗ trợ AI chứ không phải đánh giá độc lập của người chấm.

## 6. Kết luận và giới hạn

Pipeline chạy end-to-end trên PDF hiện có và so sánh có kiểm soát ba cấu hình. A được chọn theo Evidence Hit@3 dev; trên test, ba cấu hình hòa về Evidence Hit@3, còn B/C có Hit@1 cao hơn A. Rà soát hỗ trợ AI cho coherence và grounding, cùng kiểm tra metadata theo trace, đã được ghi trong các audit CSV và được phân biệt rõ với đánh giá độc lập của người chấm. Theo xác nhận của người học, PDF và trích đoạn được chia sẻ cho repo công khai; báo cáo không gán cho tài liệu một giấy phép mở không có trong nguồn.

## 7. Hỗ trợ AI và khả năng tái lập

AI hỗ trợ xây dựng pipeline, chuẩn bị benchmark, tổng hợp kết quả và rà soát coherence/grounding theo rubric. Các điểm AI-assisted được đánh dấu trong audit CSV, không trình bày như đánh giá độc lập của người chấm; metadata được đối chiếu tự động với trang nguồn. Lệnh chạy từ thư mục `starter`:

```powershell
python -m src.run --phase dev --allow-short-corpus
python -m src.run --phase test --allow-short-corpus
```

Mã pipeline ở [starter/src](../../starter/src/); benchmark, manifest, metrics, traces, stats và audit sheets được liên kết tại các mục trên. PDF được đưa vào repo theo xác nhận quyền chia sẻ của người học; tên giấy phép cụ thể không có trong tài liệu.
