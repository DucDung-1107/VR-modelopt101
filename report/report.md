 # Exercise 1 — Quantization

 ## 1. Mục tiêu và ý tưởng

 Exercise 1 giới thiệu **post-training quantization (PTQ)**: lấy một mô hình đã huấn luyện ở FP32 và biểu diễn các trọng số bằng số nguyên ít bit hơn, ở đây là INT8. Kiến trúc mạng và quá trình suy luận không thay đổi về mặt toán học; thay đổi chính là cách lưu trữ các giá trị trọng số.

 Với INT8, mỗi trọng số được ánh xạ theo công thức:

 \[
 q = \operatorname{round}(x/s) + z, \qquad
 \hat{x} = s(q-z)
 \]

 trong đó `s` là scale, `z` là zero-point, `q` là mã nguyên và `\hat{x}` là giá trị sau khi giải lượng tử hóa. Hai lựa chọn được khảo sát là:

 - **Symmetric quantization**: `z = 0`, phù hợp với trọng số thường phân bố quanh 0.
 - **Per-channel quantization**: dùng một scale cho mỗi output channel thay vì một scale cho toàn bộ tensor, nhờ đó các channel có biên độ khác nhau không làm tăng sai số của nhau.

 Mục tiêu thực nghiệm là kiểm tra sự đánh đổi giữa kích thước mô hình, độ chính xác, sai số lượng tử hóa và latency.

 ## 2. Dữ liệu và tiền xử lý

 Thí nghiệm sử dụng **MNIST**, gồm 60.000 ảnh huấn luyện và 10.000 ảnh kiểm tra. Mỗi ảnh là ảnh grayscale kích thước `28 × 28`, thuộc một trong 10 lớp chữ số viết tay từ 0 đến 9.

 Tập train được augment bằng `RandomCrop(28, padding=2)`, sau đó chuyển sang tensor và chuẩn hóa với:

 - mean = `0.1307`
 - standard deviation = `0.3081`

 Tập test chỉ được chuyển sang tensor và chuẩn hóa, không dùng random augmentation. Batch size là 128 cho train và 256 cho test.

 ## 3. Mô hình

 Mô hình là một CNN kiểu VGG đơn giản, gồm sáu convolution block. Mỗi block có `Conv2d`, `BatchNorm2d` và `ReLU`; sau mỗi hai block là một `MaxPool2d(2)`.

 Số channel tăng theo các stage:

 `1 → 64 → 64 → 128 → 128 → 256 → 256`

 Cuối mạng là `AdaptiveAvgPool2d(1)`, flatten và một fully-connected layer 10 đầu ra. Tổng số tham số là **1.147.722**. Chỉ trọng số của các lớp convolution và linear được lượng tử hóa; bias, BatchNorm parameters và running statistics vẫn giữ FP32.

 ## 4. Quy trình thực nghiệm

 Đầu tiên, mô hình FP32 được huấn luyện từ đầu trong 12 epoch bằng SGD với momentum 0.9, learning rate ban đầu 0.05, weight decay `5e-4` và cosine learning-rate schedule.

 Sau khi có baseline, notebook thực hiện các bước:

 1. Tính scale và zero-point từ miền giá trị của trọng số.
 2. Lượng tử hóa trọng số sang mã INT8 rồi giải lượng tử hóa để tạo một bản mô hình có trọng số giả lượng tử hóa.
 3. Đánh giá accuracy trên test set.
 4. Lưu các mã INT8 và scale vào packed state dict để đo kích thước trên đĩa.
 5. Đo latency suy luận một ảnh ở CPU với batch size 1.

 Ngoài kết quả trên toàn mô hình, notebook còn so sánh sai số MSE giữa per-tensor và per-channel trên trọng số convolution đầu tiên.

 ## 5. Kết quả

 ### 5.1. Baseline và INT8

 | Mô hình | Kích thước | Accuracy | CPU latency |
 |---|---:|---:|---:|
 | FP32 baseline | 4.61 MB | 99.69% | 3.90 ms |
 | INT8, symmetric per-channel | 1.18 MB | 99.69% | 3.80 ms |

 Mô hình INT8 nhỏ hơn khoảng **3.9 lần**, trong khi accuracy không giảm trong lần chạy này (`+0.00` điểm phần trăm). Mức giảm kích thước phù hợp với tỷ lệ lý thuyết `32 / 8 = 4` lần; chênh lệch nhỏ là do scale, metadata và các thành phần vẫn được lưu FP32.

 ### 5.2. Per-tensor so với per-channel

 Trên trọng số của convolution đầu tiên, kết quả MSE là:

 | Cách lượng tử hóa | INT8 MSE |
 |---|---:|
 | Symmetric per-tensor | `3.413 × 10⁻⁷` |
 | Symmetric per-channel | `1.226 × 10⁻⁷` |

 ### Per-tensor và per-channel là gì?

 Hai cách này khác nhau ở số lượng `scale` được dùng khi lượng tử hóa trọng số:

 - **Per-tensor**: toàn bộ một tensor trọng số dùng chung một `scale` và một `zero-point`. Ví dụ, một lớp convolution có trọng số dạng `[64, 1, 3, 3]` sẽ chỉ có một scale cho cả 64 output channel. Nếu một channel có giá trị lớn bất thường, scale chung phải mở rộng theo channel đó; các channel còn lại sẽ được biểu diễn thô hơn và có sai số lớn hơn.
 - **Per-channel**: mỗi output channel có một `scale` riêng. Với tensor `[64, 1, 3, 3]`, sẽ có 64 scale tương ứng với 64 filter. Mỗi filter được tận dụng gần đầy dải INT8 của nó, nên thường giảm sai số lượng tử hóa.

 Có thể hình dung per-tensor là dùng **một thước đo chung cho cả lớp**, còn per-channel là dùng **một thước đo riêng cho từng filter**. Per-channel cần lưu thêm một lượng nhỏ metadata (các scale), nhưng phần metadata này rất nhỏ so với số lượng trọng số.

 Trong lần chạy mới của bạn, accuracy là:

 | Cách biểu diễn | Accuracy | So với FP32 |
 |---|---:|---:|
 | FP32 | **99.67%** | — |
 | INT8 per-tensor | **99.66%** | -0.01 điểm phần trăm |
 | INT8 per-channel | **99.67%** | 0.00 điểm phần trăm |

 Như vậy, per-tensor chỉ giảm 0.01 điểm phần trăm và per-channel giữ nguyên accuracy. Đây là khác biệt rất nhỏ vì bài toán MNIST và model này khá dễ, đồng thời trọng số đã phân bố tương đối thuận lợi. Per-channel vẫn có ý nghĩa vì nó giảm sai số biểu diễn và thường ổn định hơn trên model/dataset khó hơn; không phải lúc nào cải thiện accuracy cũng nhìn thấy rõ ở một bài toán đơn giản.

 Về MSE, per-channel giảm MSE khoảng **64%** so với per-tensor trong phép đo trên trọng số convolution đầu tiên. Điều này xác nhận rằng các output channel có biên độ trọng số khác nhau; dùng scale riêng giúp bước lượng tử hóa sát hơn.

 Accuracy khi lượng tử hóa toàn mô hình cũng cho thấy xu hướng tương tự:

 - FP32: **99.69%**
 - INT8 per-tensor: **99.67%**
 - INT8 per-channel: **99.69%**

 Trong bài toán này, per-channel khôi phục hoàn toàn accuracy của baseline, còn per-tensor giảm nhẹ 0.02 điểm phần trăm.

 ### 5.3. Diễn giải latency

 Latency đo được gần như không đổi, ví dụ từ khoảng 3.90 ms xuống 3.80 ms. Có ba nguyên nhân chính:

 1. **Mô hình đang chạy kernel FP32**: notebook lưu mã INT8, nhưng khi suy luận lại giải lượng tử hóa thành `w_hat = scale × (q - zero_point)` rồi đưa trọng số float vào các lớp `Conv2d`/`Linear` thông thường. CPU không thực hiện phép tính convolution trực tiếp trên INT8 trong thí nghiệm này.
 2. **Đo latency không tính lợi ích lưu trữ**: latency chỉ đo thời gian forward pass trên CPU, không đo dung lượng file, thời gian đọc model từ disk hay bandwidth tiết kiệm được. Vì vậy model nhỏ hơn không tự động làm phép tính nhanh hơn.
 3. **Chi phí cố định chiếm tỷ trọng đáng kể**: một lần suy luận còn có BatchNorm, ReLU, pooling, memory movement và overhead gọi PyTorch. Việc giảm kích thước weight không loại bỏ các chi phí này.

 Muốn latency giảm rõ rệt, cần một runtime có **INT8 kernels**: tích chập/ma trận phải nhận trực tiếp mã INT8, tích lũy bằng INT32 rồi scale kết quả về dạng phù hợp. Khi đó mới giảm được số byte đọc từ bộ nhớ và có thể tận dụng instruction SIMD hoặc phần cứng tăng tốc. Các runtime triển khai như TFLite, ONNX Runtime, TensorRT hoặc backend phần cứng cụ thể mới quyết định speedup thực tế.

 Vì vậy, kết luận đúng từ thí nghiệm hiện tại là: **quantization chắc chắn giúp giảm kích thước lưu trữ; speedup chỉ xuất hiện khi deployment stack thật sự hỗ trợ integer inference**.

 ### 5.4. Bản chạy bằng INT8 kernel thật

 Để kiểm chứng phần speedup thay vì chỉ giải lượng tử hóa về FP32, notebook đã được bổ sung một pipeline FX graph-mode PTQ ở phần 12. Pipeline này:

 1. Chọn CPU quantized backend khả dụng (`x86`, `oneDNN` hoặc `fbgemm`).
 2. Chạy calibration trên một số batch để đo miền activation.
 3. Dùng `convert_fx` để thay các lớp float bằng `QuantizedConvReLU2d` và `QuantizedLinear`.
 4. Đo accuracy, kích thước state dict và latency trên CPU bằng đúng quantized operators.

 Khi chạy phần này, cần đọc latency cùng với tên backend và graph in ra. Nếu graph có `QuantizedConvReLU2d`/`QuantizedLinear`, đây là inference INT8 thực sự ở operator level. Mức speedup vẫn phụ thuộc CPU, instruction set, phiên bản PyTorch và số operator được backend hỗ trợ; không nên coi một con số đo trên một máy là mức speedup cố định cho mọi thiết bị.

 ## 6. Insight và các giả thiết được kiểm chứng thêm

 Notebook được mở rộng sau phần 10 để kiểm tra các giả thiết sau:

 1. **Giảm số bit làm tăng sai số lượng tử hóa và có thể làm giảm accuracy**: quét INT8, INT6 và INT4 trên cùng checkpoint FP32.
 2. **Per-channel hữu ích hơn khi các channel có scale khác nhau**: so sánh MSE trung bình trên tất cả convolution/linear weights giữa per-tensor và per-channel.
 3. **Affine đặc biệt hữu ích cho dữ liệu lệch khỏi 0**: tạo activation sau ReLU, rồi so sánh symmetric và affine quantization trên activation này.

 ### Kết quả kiểm chứng các hypothesis

 Kết quả chạy trên checkpoint mới:

 | Bit-width | Packed size đo được | Accuracy | Mean weight MSE |
 |---:|---:|---:|---:|
 | INT8 | 1.18 MB | 99.70% | `9.152 × 10⁻⁸` |
 | INT6 | 1.18 MB | 99.71% | `1.526 × 10⁻⁶` |
 | INT4 | 1.18 MB | 99.64% | `2.995 × 10⁻⁵` |

 Kết quả cho thấy:

 - Từ INT8 xuống INT6, MSE tăng khoảng **16,7 lần**, nhưng accuracy vẫn gần như không đổi, thậm chí lần chạy này INT6 cao hơn 0.01 điểm phần trăm. Đây là dao động nhỏ do mô hình/test set và không chứng minh INT6 tốt hơn INT8; accuracy không phải là hàm giảm đơn điệu theo MSE trong một lần chạy hữu hạn.
 - INT4 làm MSE tăng khoảng **328 lần** so với INT8 và accuracy giảm xuống 99.64%, cho thấy bắt đầu xuất hiện đánh đổi rõ ràng giữa độ chính xác biểu diễn và chất lượng mô hình.
 - Cả ba dòng đều có `packed size = 1.18 MB` vì implementation hiện tại lưu mã INT6/INT4 trong tensor `torch.int8`. Đây là kích thước container thực tế, chưa phải kích thước bit-packed. Muốn INT4 thực sự nhỏ hơn cần pack hai mã 4-bit vào một byte và cần runtime đọc format đó.

 So sánh trên toàn bộ trọng số ở INT8 cho kết quả:

 ```text
 per-tensor:   1.4840094131e-07
 per-channel:  9.1515350187e-08
 ```

 Per-channel giảm MSE khoảng **38,3%** so với per-tensor. Điều này củng cố kết luận rằng các filter có biên độ khác nhau và scale riêng giúp sử dụng dải INT8 hiệu quả hơn.

 Với activation sau ReLU:

 | Scheme | Activation MSE | Zero-point |
 |---|---:|---:|
 | Symmetric | `8.326 × 10⁻⁶` | 0.0 |
 | Affine | `2.157 × 10⁻⁶` | -128.0 |

 Affine quantization giảm MSE khoảng **74,1%**. Lý do là activation sau ReLU có miền giá trị lệch dương, với nhiều giá trị gần 0 và không có phần âm. Symmetric quantization vẫn dành một nửa dải mã cho phía âm dù activation không sử dụng miền đó; affine dịch zero-point để tận dụng toàn bộ dải INT8 cho miền dữ liệu thực tế. Đây là lý do symmetric phù hợp hơn cho weight zero-centered, còn affine thường phù hợp hơn cho activation sau ReLU.

 Các phép đo bổ sung giúp phân biệt hai loại lợi ích: bit-width quyết định giới hạn biểu diễn, còn cách chọn scale quyết định mức độ tận dụng dải số đó. Một chi tiết triển khai quan trọng là notebook đang lưu mã INT6/INT4 trong container `torch.int8`; vì vậy kích thước file đo trực tiếp có thể chưa giảm thêm. Cột kích thước lý thuyết trong cell mở rộng biểu diễn mức giảm nếu mã được bit-pack đúng định dạng triển khai. Do đó, INT8 per-channel thường là điểm cân bằng tốt cho weight-only PTQ; INT4 có thể giảm kích thước hơn nữa nhưng cần bit-packing/runtime hỗ trợ và phải kiểm tra accuracy trên từng mô hình.

 ## 7. Kết luận

 Exercise 1 cho thấy quantization là một kỹ thuật nén mô hình đơn giản nhưng hiệu quả. Trên MNIST, việc chuyển trọng số từ FP32 sang INT8 per-channel làm mô hình nhỏ hơn gần 4 lần mà không làm giảm accuracy. Per-channel cũng giảm sai số trọng số rõ rệt so với per-tensor. Các thử nghiệm bổ sung cho thấy giảm bit làm MSE tăng; INT6 vẫn giữ accuracy trong thí nghiệm này, trong khi INT4 bắt đầu giảm accuracy. Affine có lợi hơn rõ rệt trên activation sau ReLU bị lệch dương. Tuy nhiên, giảm kích thước không tự động đồng nghĩa với tăng tốc; để có speedup cần format bit-packed, runtime và kernel thực sự tính toán trực tiếp trên INT8/INT4.

 # Exercise 2 — Knowledge Distillation

 ## 1. Mục tiêu và ý tưởng

 Exercise 2 giới thiệu **knowledge distillation (KD)**: huấn luyện một model nhỏ gọi là **student** bằng cách cho nó học không chỉ từ nhãn thật, mà còn từ phân phối dự đoán mềm của một model lớn hơn gọi là **teacher**.

 Mục tiêu không phải làm student lớn hơn, mà là giữ nguyên kích thước và latency của student trong khi cải thiện accuracy. Teacher truyền thêm thông tin về mức độ giống nhau giữa các lớp. Ví dụ, thay vì chỉ nói một ảnh thuộc lớp `7`, teacher có thể cho biết ảnh đó có xác suất nhỏ thuộc `1` hoặc `9`; các xác suất tương đối này được gọi là **dark knowledge**.

 ## 2. Dữ liệu và tiền xử lý

 Thí nghiệm sử dụng **MNIST** với 60.000 ảnh train và 10.000 ảnh test. Ảnh grayscale có kích thước `28 × 28` và gồm 10 lớp chữ số viết tay.

 Train dùng `RandomCrop(28, padding=2)`, chuyển sang tensor và chuẩn hóa theo mean `0.1307`, standard deviation `0.3081`. Test chỉ chuẩn hóa, không random augmentation. Batch size là 128 cho train và 256 cho test.

 ## 3. Teacher và student

 Hai mạng nhận cùng input và dự đoán cùng 10 lớp, nhưng có capacity rất khác nhau:

 | Model | Kiến trúc chính | Tham số |
 |---|---|---:|
 | Teacher | CNN 3 stage, channel `64 → 128 → 256`, mỗi stage có 2 conv block | 1.147.722 |
 | Student | CNN nhỏ hơn, channel `16 → 32 → 32`, mỗi stage có 1 conv block | 14.458 |

 Student chỉ có khoảng **1/79,4** số tham số của teacher. Đây là model được hướng tới triển khai trên edge; teacher chỉ dùng trong quá trình training và không cần đưa vào thiết bị cuối.

 ## 4. Quy trình thực nghiệm

 Có ba model được huấn luyện cùng dataset và cùng số epoch:

 1. **Teacher**: model lớn, train bằng cross-entropy với nhãn thật.
 2. **Student baseline**: model nhỏ, train bằng cross-entropy, không dùng teacher.
 3. **Student KD**: cùng đúng kiến trúc student, nhưng loss kết hợp giữa cross-entropy và KL divergence với output của teacher.

 Loss distillation được dùng là:

 \[
 \mathcal{L} = \alpha T^2\,\mathrm{KL}​\left(\mathrm{softmax}(z_s/T)\,\|\,\mathrm{softmax}(z_t/T)\right)
 +(1-\alpha)\,\mathrm{CE}(z_s,y)
 \]

 Trong đó `T = 4` là temperature làm mềm phân phối, còn `α = 0.7` điều chỉnh mức độ tin vào teacher. Teacher được chuyển sang evaluation mode và đóng gradient; student là model duy nhất được cập nhật.

 ## 5. Kết quả

 | Model | Tham số | Kích thước | Accuracy | CPU latency |
 |---|---:|---:|---:|---:|
 | Teacher | 1.147.722 | 4.61 MB | 99.71% | 3.32 ms |
 | Student, không KD | 14.458 | 0.07 MB | 99.28% | 0.52 ms |
 | Student, có KD | 14.458 | 0.07 MB | 99.45% | 0.51 ms |

 So với student baseline, KD cải thiện accuracy từ **99.28% lên 99.45%**, tăng **0.17 điểm phần trăm**. Đồng thời:

 - số tham số không đổi: 14.458;
 - kích thước không đổi: khoảng 0.07 MB;
 - latency gần như không đổi: khoảng 0.51 ms.

 Student KD nhỏ hơn teacher **79,4 lần** và nhanh hơn khoảng **6,8 lần**, trong khi vẫn đạt accuracy cao hơn student train bằng nhãn cứng. Kết quả cho thấy chi phí lớn nằm ở training teacher; sau khi distillation xong, chỉ cần triển khai student.

 ## 6. Temperature, alpha và insight

 Temperature càng cao thì phân phối teacher càng mềm, các xác suất nhỏ của lớp không đúng càng dễ quan sát. Tuy nhiên, nếu quá cao, phân phối có thể trở nên gần uniform và tín hiệu teacher yếu đi. Hệ số `α` cũng tạo trade-off:

 - `α = 0`: student học hoàn toàn từ nhãn thật, tương đương baseline.
 - `α` lớn: student ưu tiên bắt chước teacher hơn, nhưng có thể phụ thuộc quá nhiều vào lỗi hoặc bias của teacher.
 - Giá trị trung gian như `α = 0.7` kết hợp được thông tin teacher và ground truth.

 Notebook được mở rộng để đo entropy của teacher ở nhiều temperature và tách riêng KD loss với CE loss ở nhiều alpha. Các phép đo này kiểm chứng rằng temperature cao làm tăng entropy của soft targets, còn alpha chỉ thay đổi cách phối hợp hai nguồn supervision; chúng không làm thay đổi số tham số hay latency của student.

 ## 7. Hạn chế và kết luận

 Mức cải thiện 0.17 điểm phần trăm là vừa phải vì MNIST là bài toán đơn giản và student baseline vốn đã đạt 99.28%. Trên dataset khó hơn hoặc khi student nhỏ hơn nữa, lợi ích của KD thường rõ hơn. Ngoài ra, KD chỉ chuyển kiến thức tốt nếu teacher đủ mạnh; teacher yếu hoặc sai lệch có thể truyền cả lỗi sang student.

 Exercise 2 chứng minh rằng distillation giúp tách **chi phí training** khỏi **chi phí deployment**: dùng một teacher lớn trong lúc học, sau đó triển khai student rất nhỏ. KD đặc biệt hiệu quả khi kết hợp tiếp với pruning và quantization để tạo model edge có accuracy tốt, bộ nhớ nhỏ và latency thấp.

 # Exercise 3 — Pruning

 ## 1. Mục tiêu

 Exercise 3 khảo sát pruning: loại bỏ các trọng số hoặc filter ít quan trọng để giảm chi phí mô hình. Ba yếu tố được kiểm tra là criterion, granularity và pruning ratio.

 - **Criterion**: dùng magnitude; weight nhỏ được xem là ít quan trọng, còn filter được chấm bằng L1 norm.
 - **Unstructured pruning**: xóa từng weight riêng lẻ bằng cách đặt chúng về 0.
 - **Structured pruning**: xóa toàn bộ filter/channel và rebuild mạng với width nhỏ hơn.

 ## 2. Dữ liệu, model và thực nghiệm

 Dữ liệu vẫn là MNIST 60.000/10.000 ảnh `28 × 28`, cùng augmentation và normalization như hai exercise trước. Model là CNN sáu conv block với width `64, 64, 128, 128, 256, 256`, tổng cộng **1.147.722 tham số**.

 Model dense được train 12 epoch. Sau đó thử unstructured pruning theo global magnitude và structured pruning ở ratio 50%. Structured model được fine-tune 4 epoch với learning rate nhỏ hơn để các filter còn lại phục hồi.

 ## 3. Kết quả

 Unstructured pruning không fine-tune cho kết quả:

 | Sparsity | Accuracy |
 |---:|---:|
 | 0% | 99.62% |
 | 50% | 99.60% |
 | 70% | 99.11% |
 | 80% | 87.14% |
 | 90% | 40.64% |
 | 95% | 10.10% |

 Ở 50% structured pruning:

 | Model | Size | Accuracy | CPU latency |
 |---|---:|---:|---:|
 | Dense | 4.61 MB | 99.62% | 3.37 ms |
 | Unstructured masked | 4.61 MB | 99.60% | 3.51 ms |
 | Structured + fine-tune | 1.17 MB | 99.67% | 1.83 ms |

 Structured pruning giảm khoảng **3,9 lần kích thước** và **1,8–1,9 lần latency**, trong khi accuracy sau fine-tune tăng 0.05 điểm phần trăm so với dense baseline. Accuracy 9.74% ngay sau khi rebuild là hiện tượng expected: các filter và BatchNorm statistics bị thay đổi đột ngột; fine-tuning phục hồi lại representation.

 Khi so cùng ratio 50%, L1-selected structured model đạt 99.67%, còn random-selected chỉ đạt 11.35% trước fine-tune. Điều này cho thấy criterion có ý nghĩa: L1 giữ các filter đang mang feature hữu ích, còn random có thể xóa các filter quan trọng và làm đứt representation ở nhiều stage cùng lúc. Kết quả này nên được đọc là bằng chứng của criterion magnitude trong lần chạy này, không phải guarantee cho mọi architecture.

 Insight quan trọng là 50% zero trong dense tensor chưa tạo ra 50% speedup. Muốn tận dụng unstructured sparsity cần sparse storage và sparse kernels. Structured pruning tạo tensor nhỏ thật nên backend dense thông thường cũng có thể chạy nhanh hơn.

 ## 4. Khung lý thuyết và thực hành pruning

 Pruning tối ưu một mô hình dưới constraint về số weight, channel hoặc latency. Magnitude pruning là một heuristic: nó giả định weight nhỏ có ảnh hưởng nhỏ, nhưng không đo trực tiếp độ nhạy của loss. Vì vậy quy trình thực hành cần tách rõ:

 1. train dense baseline;
 2. chọn criterion và granularity;
 3. prune một bản copy để đo accuracy ngay sau khi cắt;
 4. fine-tune để phần còn lại học bù;
 5. benchmark lại size/latency trên runtime thật.

 Với unstructured pruning, tensor shape không đổi nên dense kernel vẫn đọc và tính cả các số 0. Với structured pruning, xóa output filter buộc phải xóa input channel tương ứng ở layer sau và cập nhật BatchNorm/linear head; đây là lý do code phải rebuild toàn bộ mạng thay vì chỉ mask weight. Trong production, iterative pruning (prune ít một rồi fine-tune) thường an toàn hơn one-shot pruning ratio lớn, còn criterion tốt hơn có thể dùng Taylor sensitivity, Hessian hoặc activation statistics.

 # Exercise 4 — Neural Architecture Search

 ## 1. Mục tiêu và search space

 Exercise 4 thay việc chọn architecture bằng tay bằng một search nhỏ trên ba gene:

 - `depth ∈ {2, 3, 4}`;
 - `width ∈ {8, 12, 16, 24, 32}`;
 - `kernel ∈ {3, 5}`.

 Search space có **30 kiến trúc**. Mỗi candidate là CNN có số stage, width và kernel tương ứng. Mục tiêu là tìm model có accuracy tốt trong giới hạn size/latency phù hợp với edge.

 ## 2. Dữ liệu và search strategy

 Dữ liệu là MNIST; proxy dùng subset 10.000 ảnh và train mỗi candidate 2 epoch. Random search lấy 8 candidate. Evolutionary search tạo thêm tối đa 2 generation, mỗi generation mutate một gene của hai parent tốt nhất.

 Proxy accuracy dùng để xếp hạng nhanh, còn finalist được train đầy đủ 10 epoch trên toàn bộ train set. Search dùng model size làm cost proxy, sau đó đo lại size và CPU latency thật.

 ## 3. Kết quả

 Trong lần chạy này, baseline là `d3-w32-k3`, còn NAS pick là `d4-w12-k3`:

 | Model | Size | Accuracy | CPU latency |
 |---|---:|---:|---:|
 | Baseline `d3-w32-k3` | 0.38 MB | 99.41% | 0.95 ms |
 | NAS pick `d4-w12-k3` | 0.23 MB | 99.37% | 0.74 ms |

 NAS pick nhỏ hơn **1,7 lần**, nhanh hơn khoảng **1,3 lần**, nhưng accuracy thấp hơn **0.04 điểm phần trăm** trong lần train này. Đây là kết quả hợp lý với NAS có proxy rẻ: architecture được chọn theo accuracy sau 2 epoch trên subset, sau đó mới train lại đầy đủ; thứ hạng proxy không nhất thiết giữ nguyên sau full training.

 Output mới cũng cho thấy evolutionary search không thắng random search: best random đạt **98.14%** proxy accuracy, còn best evolved đạt **97.95%**. Correlation giữa proxy accuracy và size chỉ khoảng **+0.445**, nghĩa là model lớn thường có lợi thế nhưng quan hệ không tuyệt đối. Đây không phải lỗi; budget nhỏ, proxy chỉ train 2 epoch và kết quả phụ thuộc random seed. Vì vậy không được kết luận “evolution luôn tốt hơn random”. Kết luận đúng là evolution là một strategy có thể tập trung search quanh candidate tốt, nhưng cần nhiều budget/seed hơn để đánh giá công bằng.

 ## 4. Khung lý thuyết và thực hành NAS

 Về lý thuyết, NAS là bài toán tối ưu đa mục tiêu:

 \[
 \max_a \; \mathrm{Accuracy}(a), \qquad
 \min_a \; \mathrm{Cost}(a)
 \]

 với `a` là architecture. Vì accuracy thật của một architecture chỉ biết sau khi train, search dùng **proxy fidelity** thấp hơn: ít epoch, ít data hoặc weight sharing. Điều này giảm chi phí nhưng tạo noise và bias. Vì vậy quy trình thực hành đúng có ba tầng:

 1. định nghĩa search space và constraint phần cứng;
 2. proxy search để lọc candidate;
 3. retrain và benchmark finalist bằng workload/CPU/GPU thật.

 Trong notebook, `size` được dùng làm cost proxy vì dễ đo. Khi triển khai thực tế nên thay hoặc bổ sung cost bằng latency trên đúng batch size, memory peak, energy hoặc operator support của target device. Một NAS pick nhỏ hơn nhưng accuracy thấp hơn 0.04% như lần chạy này vẫn có thể đáng chọn nếu budget ưu tiên latency; đó là quyết định Pareto, không phải thắng tuyệt đối về accuracy.

 Các phép đo bổ sung xác nhận depth và width làm tăng parameter/size rõ rệt, trong khi kernel 5 cũng đắt hơn kernel 3. Pareto front là cách phù hợp để chọn architecture vì accuracy tối đa không phải lúc nào cũng là model tốt nhất cho thiết bị.

 # Exercise 5 — Kernel Optimization với Triton

 ## 1. Mục tiêu

 Exercise 5 giữ nguyên phép tính `C = A × B` nhưng thay đổi cách thực thi trên GPU. Kernel Triton chia output thành tile, load các mảnh của A/B, tái sử dụng dữ liệu trong on-chip memory và dùng `tl.dot` để tích lũy.

 Đây là trục tối ưu khác với quantization, KD, pruning và NAS: các exercise kia thay đổi model hoặc lượng tính toán; kernel optimization thay đổi cách hardware thực hiện cùng phép tính.

 ## 2. Thực nghiệm

 Input dùng FP16 trên GPU Tesla T4, với ma trận vuông kích thước 512, 1024, 2048 và 4096. Triton kernel được so sánh với `torch.matmul`, vốn gọi cuBLAS. Latency được đo bằng `triton.testing.do_bench`, có warm-up và synchronization phù hợp với GPU bất đồng bộ.

 Kernel được kiểm tra trước trên ma trận `1024 × 1024`. Sai số lớn nhất là **0.0625** và `torch.allclose(..., atol=0.1, rtol=0.1)` pass, nên tính đúng số học trong sai số FP16.

 ## 3. Kết quả và vấn đề cần diễn giải đúng

 Kết quả benchmark ban đầu:

 | Size | Torch ms | Triton ms | Torch TFLOP/s | Triton TFLOP/s | Triton/Torch |
 |---:|---:|---:|---:|---:|---:|
 | 512 | 0.038 | 0.233 | 7.0 | 1.2 | 0.16x |
 | 1024 | 0.067 | 1.446 | 32.0 | 1.5 | 0.05x |

 Vì vậy, kernel **đúng nhưng chưa nhanh**: average Triton/Torch chỉ khoảng **0.11x**. Nó không đạt “same speed range” với cuBLAS. Đây là kết quả hợp lý cho kernel giáo dục chưa autotune đầy đủ: thiếu tối ưu về tile shape, register/shared-memory pressure và mapping thread-to-tile. `torch.matmul` gọi vendor library đã được tối ưu nhiều năm cho Tensor Cores và T4.

 Code đã được sửa để launch kernel với `num_warps=8` và `num_stages=4`, đồng thời thêm test shape ragged và benchmark `GROUP_M=1` so với `GROUP_M=8`. Cần chạy lại trên GPU để đo tác động cụ thể; không nên tự động ghi nhận speedup nếu output mới vẫn thấp hơn cuBLAS.

 ## 4. Insight

 - Correctness trên shape chia hết không đủ; các shape lệch kiểm tra mask đọc/ghi ngoài biên.
 - `GROUP_M` không đổi output, chỉ đổi thứ tự schedule tile và khả năng reuse trong L2 cache.
 - Grouped ordering, tile size, warp count và stages là các hyperparameter của kernel; không có cấu hình tốt nhất cho mọi GPU/shape.
 - Một kernel custom đạt đúng kết quả nhưng chậm hơn vendor kernel vẫn là kết quả có giá trị: nó cho thấy tối ưu GPU không chỉ là viết tiling, mà còn cần autotuning và tận dụng đúng phần cứng.

 ## 4. Khung lý thuyết và thực hành kernel optimization

 Với matmul, chi phí lý thuyết là `2MNK` FLOPs, nhưng thời gian thực tế còn phụ thuộc vào:

 - **Arithmetic intensity**: mỗi byte được load lên chip phục vụ bao nhiêu phép tính;
 - **memory coalescing**: các thread có đọc vùng nhớ liên tiếp hay không;
 - **occupancy và register pressure**: tile lớn tăng reuse nhưng có thể dùng quá nhiều register;
 - **Tensor Core path**: dtype, tile shape và warp configuration phải phù hợp để phần cứng chạy đúng đường tăng tốc;
 - **launch/scheduling overhead**: đặc biệt rõ ở ma trận nhỏ.

 Tiling và grouped ordering giảm global-memory traffic, nhưng không tự động đánh bại cuBLAS. Khung thực hành nên là: (1) xác nhận correctness trên shape chia hết và ragged; (2) đo bằng timer có synchronize/warm-up; (3) sweep block size, `num_warps`, `num_stages`, `GROUP_M`; (4) so sánh throughput, latency và sai số; (5) chỉ kết luận speedup trên đúng GPU và workload mục tiêu. Trong output mới, shape ragged có max error 0.0002–0.0156 và `GROUP_M=8` đạt 0.708 ms so với 0.717 ms của `GROUP_M=1`, cho thấy mask đúng và grouped ordering có lợi nhỏ ở shape 1024; nhưng lợi ích này chưa đủ bù khoảng cách với cuBLAS.
