# Day 1

<br>

# 簡立峰

- 最好的時代最壞的時代
- 機會在跨領域
- 機會在台灣的周遭

# 紀懷新
## Optimizing for User Experience with Data Science and Machine Learning
- 人與人之間, 人與機器之間 要有什麼樣的互動
- 是否有在push forward 社會
- improve user experience
    - Measure -> optimize -> impact

- social media 
    - 建立人與人connection 的 tool
    - 語言的隔閡
    - 大部分的人都follow 最popular 的人
    - different between twitter & facebook, 做朋友需要別人同意
    - Why did you wnat to climb MountEverest?
        - Because it's there.

    - 目的
        - connecting people across the word
        - challenge: 
            - Connect across lumpy 
            - both structurally and across language barriers.


    - different social medie 有不同的結果(國家)
    - 如何真正conn user?
        - AI

- The Recommendation Problem
    - user + context = item 
        - very hard.
    
    - Use RNN
        - RNN model with time.
        - attention mechanism

    - We don't represent user/items equally
    - Force learning 
        - Force selection
        - Approach
        - High-Level approach

        - force group & unforce group

- The Lesson
    - major the problem is work
    - Optimization in the real world is complex

# 邱中鎮
## Recent advances of deep learning in Google

- Image classification

- Start with Alex-net
- ResNet - Deep Residual Learn
    - 解決問題:
        - 層數越深, 效果不一定越好

- Learning the architecture
    - learn to learn

- Dealing with squentatial data 
    - RNN
    - LSTM
    - sequence to sequence with attention
    - Neural machine translation
    - Attention is all you need

- hearlthcare

- Robotics - learning from demonstration

- Art

- Tensorflow

- Eager Execution

- 殺雞焉用牛刀，某些場景不需要deep learning.



# Romeo Kienzler

## Realtime-Cognitive IoT using Deep Learning and Online Learning


- data parallelism
- intra-model parallelism


- Apache Spark
- TensorflowSPark
- Caffe on Spark


- CS229


# 宋政隆

## 深度學習環境建置與模型訓練實務


- GKE
- ETcd

- 解決多GPU, 多機器的問題
- salt cloud


# 莊永裕
## Deep Learning for Computational Photography

- photography

    - camera
    - scene
    - photographer

    - 處理各種不完美
    - `調整照片的成像`

- Demosaicking

- image signal processing

    - supervised
    - unsupervised


# 陳彥呈
## 影音大數據商機挖掘

- Viscovery = Video Discovery
- 好的內容，不僅限於原生內容, 好的聚合與推薦同樣有價值
- 好的機會好的內容有機會存活在googel & facebook 的規則之外
- 高維度的詛咒
    - 變數量往往遠大於樣本量
    - 機器學習無解
    solution:
        - 各種角度的明星臉都要辨識
        - 搭配爬蟲
- 抽象性概念例如情緒難以辨別(主觀因素)
- 需要建立反饋機制


---

# Day2 

<br>

# 黃士傑
## AlphaGo

- Solve intelligence. Use it to make the world a better place.

- Demis Hassabls 


- 人類下圍棋的直覺，策略網路(Policy Network)
- Value network
    - 價值網路
    - 判斷形勢
    - 左右互搏的自我學習
    - 克服overfitting

- TPU
- Dual Network(20-block ResNet)
- 改進training pipeline

- ### AlphaGoZero
- RL training pipeline - continuous pipeline
- remove rollouts, handcarfted features


# 吳毅成
## CGI and CGI

- First CGI
    - `Computer Games and Intelligenci`
    - 簡稱CGI lab
    - 西洋棋之餘AI，相當於果蠅之餘基因
    - 六子棋
    - CGI-2048
    - 象棋, Chimo
    
    - ways
        - way 1
            - Alpha-beta search
            - Monte-Carlo tree search(MCTS)
        - way 2
            - Machine learning
            - Learning network
    - Reinforcement learning(RL), 強化學習
        - Agent -> environment -> Agent -> environment
        - Moto-Carlo learning
            - Unbiased, but high variance
            - 類似統計的概念, 看哪個選項比較好
        - Temporal-Difference(TD) learning
            - Biased, but lower varience
    
    - 2048
        - TD learning
        - 對每個盤面評估分數
        - CNN not work
        - Need to use value function approximator
        - 遊戲的規則可以改變, model 會自動學習
        - N-Tuple Network
        - Multi-Stage TD learning 
    
    - Connect6
        - TD learning
    - 象棋
        - Comparison turning
        - n-tuple network are used too

- Second CGI
    - `CGI Go Intelligence`
    - value notwork
    - DCNN
    - one GPU and six CPU


# 林軒田
## The Interplay of Big Data, Machine Learning, and Artificial Intelligence

- ### What is `Artificial intelligence`?

- ### What is `Machine learning`?

- ### What is `Big data` ?

- 導航, Bigger data towards better AI 
    - 預測路徑的方式
        - best path by shortest distance
        - best path by currect traffic
        - best path by predicted travel time.
            - 預測抵達的time
                - 需要多樣性的資料

- From data to ML
    - Academia
        - Many acasemic projects based on embarassingly small data.
        - ML not specific to bug data: small-data analysis also important 

    - Industy
        - System much more important than model
        - Need big help from domain knowledge
        - Proper procedure very important

- From ml to ai
    - Bottom-up from ml to ai: ml tech given, what ai can do it
    - `Dream your own target` -- yes, that was me, as ML researcher.

    industy:
        - Top-down approach, what mt tech to use?
        - Resource constraints very important
        - A black plum is as sweet as a white: Any useful model is good

- From bigdata to ai
    - why not?
    - equally vaable route: big data -> human learn -> ai
    - Statistical analysis: without significance test can be useful

- AI 在意的是菜好不好吃
    - 最後好不好用

- Has big data made ai easier?
    - possibl,  but `easier than what?`
    - 設定有用的目標, 適當的難度, 比較的對象

- ML must learn to make ai easier  
    - 每年都在改變
    - 2016
        - **simple model** (important)
        - feature processing
        - complexity control
        - model selection
    - 2017
        - simple model +
        - feature processing (based on DL)
        - complexity control ++
        - model selection (properly and systematically)

- 雜項
    - System is important
        - work with system
    - When not to use ML as first choice?
    - Your favorite ML toolbox(掌握自己的coding skill)

# 李宏毅
## GAN

- when do i need generation
    - f: x -> y
    - type of machine learning 
        - regression
        - classification
        - structured learning / prediction
            - `machione has to know to gen structured object`

- generation
    - basic idea of GAN
        - generator(NN)
            - input vector, output matrix
            - Student
        - Discriminator
            - input matrix, output scalar(larger value means real)
            - teacher
    
    - Algorithm
        - init generator & Discriminator 
        - in each training iteration
            - learn D
            - learn G

    - Gen image
    - Gen 唐詩

- conditinoal generation
    - 根據指示來產生
    - text to image
    - Two type of Discriminator
        - input two things: 
            - text(input of generator)
            - image(output of generator)
    - Chat-bot with GAN
    - Speech Enhancement

- unsupervides conditinol generation
    - Cycle GAN
        - Use two generator
        - positive to nagative (NLP)
    - Disco GAN(same to Cycle GAN)

    - Abstractive Summarization
        - Seq2Seq * 2
            - 長文 -> 短文 -> 原來的長文
        - 1 * Discriminator
            - 判斷摘要是否人寫的句子

- Reinforcement learning 


# 陳縕儂
## 深度學習之智慧型對話機器人 Deep Learning for Intelligent Conversational Bots

- intelligent assistant
    - Why we need
        - ironman's home
    - Dialogue System, 對話系統
    
    - App -> Bot
        - A bot is responsible for a `single-domain`, similar to an app.
        - GUI vs CUI(Conversational UI)

    - Task-Oriented, Chit-Chat
        - 發展不同
        - 架構不同

- paper, Task-oriented Dialogue System, Young, 2000 
    - Rule based Management
        - state tracking



    







    

    


















