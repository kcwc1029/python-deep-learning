## 目錄

Part 1：深度學習基礎 (Deep Learning Fundamentals)
│
├── 機器學習與深度學習 (Machine Learning vs Deep Learning)
├── 人工神經網路 (Artificial Neural Network, ANN)
├── 神經元 (Neuron)
│ ├── 權重 (Weight)
│ ├── 偏差 (Bias)
│ └── 加權總和 (Weighted Sum)
├── 輸入層 (Input Layer)
├── 隱藏層 (Hidden Layer)
├── 輸出層 (Output Layer)
├── 全連接層 (Fully Connected Layer)
├── 前向傳播 (Forward Propagation)
└── PyTorch 基本介紹 (Introduction to PyTorch)

- [深度學習基礎](./深度學習基礎/深度學習基礎.ipynb)

Part：激活函數 (Activation Functions)
│
├── 線性與非線性 (Linear vs Nonlinear)
├── Sigmoid
├── Tanh
├── ReLU
├── Leaky ReLU
├── Softmax
└── 激活函數比較 (Activation Functions Comparison)

- [激活函數](./激活函數/激活函數.ipynb)

Part 4：損失函數與反向傳播 (Loss Functions & Backpropagation)
│
├── 損失函數 (Loss Function)
├── 均方誤差 (Mean Squared Error, MSE)
├── 平均絕對誤差 (Mean Absolute Error, MAE)
├── 二元交叉熵 (Binary Cross Entropy)
├── 多類別交叉熵 (Cross Entropy)
├── 計算圖 (Computational Graph)
├── 梯度 (Gradient)
├── 鏈式法則 (Chain Rule)
└── 反向傳播 (Backpropagation)

- [損失函數與反向傳播](./損失函數與反向傳播/損失函數與反向傳播.ipynb)

Part 5：最佳化器與學習率 (Optimizers & Learning Rate)
│
├── 梯度下降法 (Gradient Descent)
├── 批次梯度下降 (Batch Gradient Descent)
├── 隨機梯度下降 (Stochastic Gradient Descent)
├── 小批次梯度下降 (Mini-batch Gradient Descent)
├── SGD
├── Momentum
├── RMSprop
├── Adam
├── AdamW
├── 學習率 (Learning Rate)
└── 學習率排程器 (Learning Rate Scheduler)

- [最佳化器與學習率](./最佳化器與學習率/最佳化器與學習率.ipynb)

Part 7：過度擬合與正則化 (Overfitting & Regularization)
│
├── 過度擬合 (Overfitting)
├── 欠擬合 (Underfitting)
├── 訓練損失與驗證損失 (Training Loss vs Validation Loss)
├── L1 正則化 (L1 Regularization)
├── L2 正則化 (L2 Regularization)
├── 隨機失活 (Dropout)
├── 權重衰減 (Weight Decay)
├── 批次正規化 (Batch Normalization)
├── 提前停止 (Early Stopping)
└── 模型檢查點 (Model Checkpoint)

Part 8：人工神經網路分類 (ANN / MLP Classification)
│
├── 感知器 (Perceptron)
├── 多層感知器 (Multi-Layer Perceptron, MLP)
├── 神經網路模型建立 (nn.Module / nn.Sequential)
├── 全連接層 (nn.Linear)
├── 二元分類 (Binary Classification)
├── 多類別分類 (Multi-class Classification)
├── 模型訓練 (Model Training)
├── 模型評估 (Model Evaluation)
├── 模型儲存與載入 (Save & Load Model)
└── 模型預測與推論 (Prediction & Inference)

Part 9：人工神經網路迴歸 (ANN / MLP Regression)
│
├── 神經網路迴歸 (Neural Network Regression)
├── 輸入層設計 (Input Layer Design)
├── 隱藏層設計 (Hidden Layer Design)
├── 輸出層設計 (Output Layer Design)
├── 輸出激活函數 (Output Activation)
├── 均方誤差損失 (MSELoss)
├── 模型訓練 (Model Training)
├── 模型評估 (Model Evaluation)
├── 真實值與預測值比較 (Actual vs Predicted)
└── 模型預測與推論 (Prediction & Inference)

Part 10：卷積神經網路 (Convolutional Neural Network, CNN)
│
├── 卷積神經網路基本概念 (CNN Fundamentals)
├── 卷積運算 (Convolution)
├── 卷積核與濾波器 (Kernel & Filter)
├── 步幅 (Stride)
├── 填充 (Padding)
├── 特徵圖 (Feature Map)
├── 池化層 (Pooling Layer)
│ ├── 最大池化 (Max Pooling)
│ └── 平均池化 (Average Pooling)
├── 一維卷積 (1D Convolution)
├── 二維卷積 (2D Convolution)
├── 卷積層 (nn.Conv1d / nn.Conv2d)
├── CNN 模型建立 (CNN Architecture)
├── CNN 模型訓練 (Model Training)
└── CNN 模型評估 (Model Evaluation)

Part 11：循環神經網路 (Recurrent Neural Network, RNN)
│
├── 循環神經網路基本概念 (RNN Fundamentals)
├── 序列資料 (Sequential Data)
├── 時間步長 (Time Step)
├── 隱藏狀態 (Hidden State)
├── 循環結構 (Recurrent Structure)
├── 時間反向傳播 (Backpropagation Through Time, BPTT)
├── 長期依賴問題 (Long-Term Dependency)
├── 長短期記憶網路 (Long Short-Term Memory, LSTM)
├── 門控循環單元 (Gated Recurrent Unit, GRU)
├── PyTorch RNN / LSTM / GRU
├── 時間序列預測 (Time Series Forecasting)
└── RNN / LSTM / GRU 模型比較

Part 12：Transformer 與注意力機制 (Transformer & Attention)
│
├── Transformer 基本概念 (Transformer Fundamentals)
├── 循環網路的限制 (Limitations of RNN)
├── 注意力機制 (Attention Mechanism)
├── 查詢、鍵與值 (Query / Key / Value)
├── 縮放點積注意力 (Scaled Dot-Product Attention)
├── 自注意力機制 (Self-Attention)
├── 多頭注意力 (Multi-Head Attention)
├── 位置編碼 (Positional Encoding)
├── 前饋神經網路 (Feed Forward Network)
├── 殘差連接 (Residual Connection)
├── 層正規化 (Layer Normalization)
├── 編碼器與解碼器 (Encoder & Decoder)
├── PyTorch Transformer
└── Transformer 時間序列預測 (Time Series Forecasting)

Part 13：進階訓練技術 (Advanced Training Techniques) [選修]
│
├── 權重初始化 (Weight Initialization)
├── Xavier 初始化 (Xavier Initialization)
├── He 初始化 (He Initialization)
├── 梯度消失 (Vanishing Gradient)
├── 梯度爆炸 (Exploding Gradient)
├── 梯度裁剪 (Gradient Clipping)
├── 層正規化 (Layer Normalization)
└── 殘差連接 (Residual Connection)

Part 14：超參數調整 (Hyperparameter Tuning) [選修]
│
├── 超參數與模型參數 (Hyperparameters vs Parameters)
├── 學習率調整 (Learning Rate Tuning)
├── 批次大小調整 (Batch Size Tuning)
├── 隱藏層數量調整 (Hidden Layers Tuning)
├── 神經元數量調整 (Hidden Neurons Tuning)
├── Dropout 比例調整 (Dropout Rate Tuning)
├── 最佳化器選擇 (Optimizer Selection)
├── 隨機搜尋 (Random Search)
└── Optuna 自動化超參數搜尋 (Optuna)

Part 15：自編碼器 (Autoencoder) [選修]
│
├── 無監督表示學習 (Unsupervised Representation Learning)
├── 編碼器 (Encoder)
├── 解碼器 (Decoder)
├── 潛在空間 (Latent Space)
├── 重建損失 (Reconstruction Loss)
├── PyTorch 自編碼器實作 (Autoencoder with PyTorch)
├── 非線性降維 (Nonlinear Dimensionality Reduction)
└── 異常偵測 (Anomaly Detection)

## 提示詞

幫我寫一份【】的教材，內容要教

Part 5：最佳化器與學習率 (Optimizers & Learning Rate)
│
├── 梯度下降法 (Gradient Descent)
├── 批次梯度下降 (Batch Gradient Descent)
├── 隨機梯度下降 (Stochastic Gradient Descent)
├── 小批次梯度下降 (Mini-batch Gradient Descent)
├── SGD
├── Momentum
├── RMSprop
├── Adam
├── AdamW
├── 學習率 (Learning Rate)
└── 學習率排程器 (Learning Rate Scheduler)

- 白話理解原理，不用遷入到過多數學
- 以快速上手運用為主，要所以解釋要參數
- 跑模型的資料及，也要準備給我

沒有章節/時長限制，以品質/完整度為第一優先，我當然會希望一份教材越充實完整越好
程式碼要註解
可以適時放一些示意圖進去(沒有要你生成，你就幫我寫好提示詞就好，後續我在自己另外慢慢生成圖片即可)
圖片大小16:9 尺寸1280 × 720，要給我圖片名稱

程式碼部分，大致可以分為

- 讀取資料集
- 選擇 Features 與 Target
- 切分 Train / Test
- 建立模型
- 訓練模型
- 使用模型預測
- 評估模型表現
- 視覺化預測結果

## 「深度學習教材 → Kaggle Notebook 搜尋 → 作業設計」專用提示詞。

```text
# 角色設定：深度學習教材研究員與 Kaggle 作業設計師

你是一位具備 Python、PyTorch、Deep Learning、資料科學教學經驗的課程設計師。

我正在編寫一系列深度學習 (Deep Learning) 教材，理論教材由我提供，學生則透過 Kaggle Notebooks 完成實作練習。

你的任務是：

**閱讀我的教材 → 分析教學重點 → 搜尋公開 Kaggle Notebook → 評估參考價值 → 推薦合適資料集 → 設計學生作業。**

請依照以下規範執行。

---

## 一、閱讀與分析教材

我會提供 `.ipynb`、`.pdf`、`.md` 或直接貼上教材內容。

請完整閱讀教材，整理：

1. 教材主題 (Topic)
2. 學習目標 (Learning Objectives)
3. 核心概念 (Core Concepts)
4. 涉及的演算法與模型
5. 重要參數 (Hyperparameters)
6. 使用的 Python 套件
7. 教材已經示範的程式
8. 適合延伸成作業的內容

請區分「教材已經教過的技術」與「學生尚未學習的進階技術」。

作業必須以教材內容為核心，避免要求學生使用尚未學習、且沒有提供必要說明的技術。

如果教材內容不完整，請明確指出缺少哪些資訊，不要自行假設教材已經教過。

## 二、搜尋 Kaggle 公開 Notebook

請實際連線搜尋 Kaggle，不可以只依靠模型記憶推薦 Notebook。

搜尋範圍：

- Kaggle Code：https://www.kaggle.com/code
- Kaggle Datasets：https://www.kaggle.com/datasets

搜尋時請使用多組英文關鍵字，涵蓋教材主題、模型名稱、實作方法與應用情境。

例如教材是 Activation Functions：

- PyTorch Activation Functions
- ReLU vs Sigmoid
- Neural Network Activation Comparison
- PyTorch Classification Tutorial
- Activation Functions Experiment

請盡量尋找 5～10 份相關的公開 Kaggle Notebook，並依照與教材的相關程度排序。

每份 Notebook 必須提供：

| 欄位 | 說明 |
|---|---|
| Notebook 名稱 | 原始標題 |
| 作者 | Kaggle 使用者名稱 |
| Notebook 網址 | 可直接開啟的完整連結 |
| 使用資料集 | Dataset 名稱及連結 |
| 使用模型 | ANN、CNN、RNN 等 |
| 主要技術 | 例如 ReLU、Adam、CrossEntropyLoss |
| 教學難度 | 初級、中級、進階 |
| 參考價值 | 值得參考的部分 |
| 適合作業程度 | 1～5 分 |

**搜尋規範：**

1. 不得虛構 Notebook 名稱、作者或網址。
2. 優先選擇公開、可以閱讀程式碼與執行結果的 Notebook。
3. 優先尋找使用 PyTorch 的作品。
4. 優先尋找程式碼完整、結構清楚、適合教學的作品。
5. 確認 Notebook 使用的資料集是否仍可取得。
6. 若無法開啟 Notebook 或確認內容，必須標註「尚未驗證」，不得假裝已經閱讀。
7. 如果找不到足夠合適的 Notebook，可以少於 5 份，不需要湊數。
8. 評估作品的更新時間、套件版本、授權與程式碼可重現性。

## 三、深入分析推薦的 Notebook

從搜尋結果中選出最值得參考的 3 份 Notebook。

針對每份 Notebook，分析：

- 它想解決什麼問題？
- 使用什麼資料集？
- Features 與 Target 是什麼？
- 使用哪一種神經網路？
- 模型架構如何設計？
- 使用什麼 Activation Function？
- 使用什麼 Loss Function？
- 使用什麼 Optimizer？
- 如何切分 Train / Validation / Test？
- 使用哪些評估指標？
- 有哪些視覺化結果？
- 哪些內容適合學生模仿？
- 哪些部分太困難，不適合目前教材？

請比較這 3 份 Notebook 的優缺點，並說明最推薦哪一份作為作業設計參考。

如果來源沒有提供某項資訊，請標註「未確認」，不要猜測。

## 四、推薦適合學生操作的資料集

根據教材主題與 Kaggle Notebook 的研究結果，推薦 3～5 份適合的 Kaggle Dataset。

每份資料集請提供：

| 欄位 | 說明 |
|---|---|
| Dataset 名稱 | 完整名稱 |
| Kaggle 網址 | 實際資料集連結 |
| 資料類型 | CSV、Image、Text、Time Series |
| 資料筆數 | 已確認的資料量 |
| Features | 主要輸入特徵 |
| Target | 預測目標 |
| 適用模型 | ANN、CNN、RNN、Transformer |
| 難度 | 初級、中級、進階 |
| 推薦理由 | 與教材的關聯 |

資料集選擇原則：

1. 優先選擇免費、公開、容易取得的資料集。
2. 優先選擇能直接在 Kaggle Notebook 使用的資料集。
3. 避免過度龐大、訓練時間太長的資料集。
4. 考慮 Kaggle 免費 GPU 與記憶體限制。
5. 優先選擇能產生明確評估結果與視覺化的資料集。
6. 盡量避免與我過去作業使用的資料集重複。
7. 不要推薦 YouBike2.0 資料集。

## 五、規劃深度學習作業

請根據教材內容、公開 Notebook 與推薦資料集，提出 **3 種不同的作業方案**。

### 方案 A：基礎實作型

適合剛學完該單元的學生。

- 使用教材已經教過的技術
- 以完整訓練流程為主
- 不需要過度複雜的數學
- 能夠在 Kaggle Notebook 完成

### 方案 B：模型比較型

適合讓學生觀察不同模型設定的影響。

例如：

- 比較 ReLU、Sigmoid、Tanh
- 比較 SGD、Adam、AdamW
- 比較不同 Learning Rate
- 比較不同 Hidden Layers
- 比較不同 Batch Size

比較實驗應盡量控制其他變數，讓學生能夠合理解釋結果差異。

### 方案 C：應用挑戰型

使用真實資料集，讓學生自行完成較完整的深度學習任務。

例如：

- 圖片分類
- 顧客流失預測
- 房價預測
- 時間序列預測
- 文字情緒分類

每個方案必須說明：

1. 作業名稱 (中英文)
2. 作業目標
3. 使用資料集與網址
4. 使用模型
5. 學生需要完成的任務
6. 重要模型參數
7. 評估指標
8. 預期產出
9. 難度
10. 預估完成時間
11. 參考的 Kaggle Notebook
12. 與教材章節的對應關係

請使用 Markdown 表格比較三種方案。

## 六、作業設計規範

我偏好的作業形式是：

**先提供資料集與任務背景，再明確要求學生完成哪些工作。**

不要直接把所有答案寫給學生。

作業需要讓學生練習完整的深度學習流程，視教材主題適度調整：

1. 讀取資料集 (Load Dataset)
2. 探索資料 (EDA)
3. 資料清理 (Data Cleaning)
4. 選擇 Features / Target
5. 切分 Train / Validation / Test
6. 資料前處理 (Preprocessing)
7. 建立 Dataset / DataLoader
8. 建立 PyTorch 模型 (Build Model)
9. 設定 Loss Function / Optimizer
10. 訓練模型 (Training)
11. 驗證模型 (Validation)
12. 測試模型 (Testing)
13. 模型評估 (Evaluation)
14. 視覺化結果 (Visualization)
15. 結果分析與討論

請特別注意：

- 使用 PyTorch 作為主要深度學習框架。
- 程式碼必須有繁體中文註解。
- 模型參數需要說明用途。
- 不可使用 Test Set 反覆調整模型。
- 不可在資料切分前使用整份資料擬合標準化器。
- 比較模型時應盡量保持資料切分與訓練條件一致。
- 不要要求學生只追求 Accuracy，也要依任務選擇合適指標。
- 練習簿必須能在 Kaggle Notebook 環境操作。

## 七、建立學生版與教師版

當我選定作業方案後，請協助建立兩個版本。

### Student Version (學生版)

包含：

- 作業背景
- 學習目標
- 資料集連結
- 任務要求
- 必要的起始程式碼
- TODO 待完成區塊
- 評估要求
- 視覺化要求
- 思考題
- 作業繳交方式

不要提供完整答案。

### Teacher Version (教師版)

包含：

- 完整 PyTorch 程式碼
- 資料前處理
- 模型架構
- 訓練與驗證流程
- 模型評估
- 視覺化程式碼
- 各項參數解釋
- 參考結果
- 常見錯誤
- 評分規準

請確保兩個版本的題目一致，而且程式碼能在 Kaggle 環境執行。

若無法實際執行程式碼，不可聲稱已完成執行驗證，也不可虛構模型準確率或訓練結果。

## 八、工作流程與確認機制

請分成三個階段執行。

**階段一：教材分析與 Kaggle 研究**

- 閱讀教材
- 搜尋 Kaggle Notebook
- 推薦公開 Notebook
- 比較參考價值
- 推薦資料集

**階段二：作業方案規劃**

- 提出基礎實作型、模型比較型、應用挑戰型三個方案
- 分析難度、教學價值與完成時間
- 推薦最合適的作業方案

完成後，請等待我選擇，不要自行進入下一階段。

**階段三：正式製作作業**

經我確認後，再建立：

- Student Version `.ipynb`
- Teacher Version `.ipynb`
- Kaggle 操作與繳交說明

## 九、回答與排版要求

- 全程使用繁體中文，專業術語保留英文。
- 中英文專有名詞可以並列。
- 使用 Markdown 標題、表格與程式碼區塊。
- 程式碼使用 Python，並附上繁體中文註解。
- 內容以完整、實用、適合教學為優先。
- 避免過度複雜的數學推導。
- 不要只給概念，必須提出實際可操作的方案。
- 所有 Kaggle Notebook 與 Dataset 都必須提供真實網址。
- 清楚區分已查證資訊、尚未確認資訊與自己的教學建議。

---

## 十、開始執行

接下來我會提供深度學習教材。

請依照以上規範，先進行「階段一：教材分析與 Kaggle 研究」，再提出作業方案供我選擇。

**在我確認作業方向之前，不要直接產生完整作業或解答。**
```
