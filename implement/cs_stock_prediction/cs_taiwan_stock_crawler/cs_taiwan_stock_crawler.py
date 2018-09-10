# -*- coding: utf-8 -*-
from twstock import Stock
import numpy as np

class cs_taiwan_stock_crawler:
    def __init__(self):
        pass
    
    def crawl_samples(self):
        # stock number
        stock = Stock('6188')
        
        # fetch from
        stock.fetch_from(2016, 9)
        
        # price
        price = stock.close
        # 5 days ma
        ma_5 = stock.moving_average(stock.price, 5)
        # 10 days ma
        ma_10 = stock.moving_average(stock.price, 10)
        # 20 days ma
        ma_20 = stock.moving_average(stock.price, 20)
        # 60 days ma
        ma_60 = stock.moving_average(stock.price, 60)
        # 120 days ma
        ma_120 = stock.moving_average(stock.price, 120)
        # 240 days ma
        ma_240 = stock.moving_average(stock.price, 240)
        # 5 days ma-bias
        ma_bias_5 = stock.ma_bias_ratio(1, 5)
        # 10 days ma-bias
        ma_bias_10 = stock.ma_bias_ratio(1, 10)
        # 20 days ma-bias
        ma_bias_20 = stock.ma_bias_ratio(1, 20)
        
        #print(stock.date)
        #print("len(price):", len(price))
        #print("price:", price)
        #print("len(ma_5):", len(ma_5))
        #print("ma_5:", ma_5)
        #print("len(ma_10):", len(ma_10))
        #print("len(ma_20):", len(ma_20))
        #print("len(ma_60):", len(ma_60))
        #print("len(ma_120):", len(ma_120))
        #print("len(ma_240):", len(ma_240))
        #print("len(ma_bias_5):", len(ma_bias_5))
        #print("len(ma_bias_10):", len(ma_bias_10))
        #print("len(ma_bias_20):", len(ma_bias_20))
        
        # settings
        sample_num = 2
        day_of_prediction = 20
                
        if (len(ma_240) - day_of_prediction - sample_num) < 0 :
            print("error : len(ma_240) is too little")
            return ( np.array([]), np.array([]) )
        
        # _get labels
        #_labels = price[len(price) - sample_num : len(price)]
        #print("_len(_labels):", len(_labels))
        #print(_labels)
            
        # get labels and samples
        labels = []
        samples = []
        for sample_idx in range(0 , sample_num):
            reverse_sample_idx = sample_num - sample_idx - 1
            #print(reverse_sample_idx)
            
            label = []
            label.extend( price[(len(price) - reverse_sample_idx - 1) : (len(price) - reverse_sample_idx)] )
            labels.append(label)
            
            sample = []
            sample.extend( price[ (len(price) - day_of_prediction - reverse_sample_idx - 1) : (len(price) - day_of_prediction - reverse_sample_idx) ] )
#            sample.extend( ma_5[ (len(ma_5) - day_of_prediction - reverse_sample_idx - 1) : (len(ma_5) - day_of_prediction - reverse_sample_idx) ] )
#            sample.extend( ma_10[ (len(ma_10) - day_of_prediction - reverse_sample_idx - 1) : (len(ma_10) - day_of_prediction - reverse_sample_idx) ] )
#            sample.extend( ma_20[ (len(ma_20) - day_of_prediction - reverse_sample_idx - 1) : (len(ma_20) - day_of_prediction - reverse_sample_idx) ] )
#            sample.extend( ma_60[ (len(ma_60) - day_of_prediction - reverse_sample_idx - 1) : (len(ma_60) - day_of_prediction - reverse_sample_idx) ] )
#            sample.extend( ma_120[ (len(ma_120) - day_of_prediction - reverse_sample_idx - 1) : (len(ma_120) - day_of_prediction - reverse_sample_idx) ] )
#            sample.extend( ma_240[ (len(ma_240) - day_of_prediction - reverse_sample_idx - 1) : (len(ma_240) - day_of_prediction - reverse_sample_idx) ] )
            sample.extend( ma_bias_5[ (len(ma_bias_5) - day_of_prediction - reverse_sample_idx - 1) : (len(ma_bias_5) - day_of_prediction - reverse_sample_idx) ] )
#            sample.extend( ma_bias_10[ (len(ma_bias_10) - day_of_prediction - reverse_sample_idx - 1) : (len(ma_bias_10) - day_of_prediction - reverse_sample_idx) ] )
#            sample.extend( ma_bias_20[ (len(ma_bias_20) - day_of_prediction - reverse_sample_idx - 1) : (len(ma_bias_20) - day_of_prediction - reverse_sample_idx) ] )
            samples.append(sample)
            
        print("len(labels):", len(labels))
        print(labels)
        
        print("len(samples):", len(samples))
        print(samples)
        
        #return ( labels, samples )
        return ( np.array(labels), np.array(samples) )
