#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .gnn_model import GNNModel
from .transformer_model import TimeSeriesTransformer
from .multi_task_model import MultiTaskModel

class UltraAdvancedSalesForecastModel(nn.Module):
    """ 
    Args:
        num_skus (int): Количество SKU.
        num_regions (int): Количество регионов.
        embedding_dim (int): Размерность эмбеддингов.
        hidden_dim (int): Размерность скрытых слоев.
        output_dim (int): Размерность выхода.
        num_layers (int): Количество слоев трансформера.
        seasonality_period (int): Период сезонности.
    """
    def __init__(self, num_skus, num_regions, embedding_dim, hidden_dim, output_dim, num_heads, num_layers, seasonality_period):
        super(UltraAdvancedSalesForecastModel, self).__init__()
        
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim и hidden_dim должны быть положительными.")
        
        # Встраивания
        self.sku_embedding = nn.Embedding(num_skus, embedding_dim)
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)
        
        # Иерархические модели
        self.sku_model = SKUModel(embedding_dim, hidden_dim)
        self.region_model = RegionModel(embedding_dim, hidden_dim)
        
        # Трансформер
        self.transformer = TimeSeriesTransformer(input_dim=hidden_dim * 2 + 1, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
        
        # Внимание
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        # Слои для тренда, сезонности и доверительных интервалов
        self.trend_layer = TrendLayer(hidden_dim, hidden_dim // 2)
        self.seasonality_layer = SeasonalityLayer(hidden_dim, hidden_dim // 2, seasonality_period)
        self.confidence_layer = ConfidenceIntervalLayer(hidden_dim, hidden_dim // 2)
        
        # Полносвязные слои
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 1 + seasonality_period, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, sku, region, sales_history):
        """
        Forward pass для модели.
        
        Args:
            sku (torch.Tensor): Индексы SKU.
            region (torch.Tensor): Индексы регионов.
            sales_history (torch.Tensor): Исторические данные о продажах.
        
        Returns:
            tuple: Прогноз продаж, тренд, сезонность, доверительные интервалы.
        """
        # Встраивания
        sku_embedded = self.sku_embedding(sku)
        region_embedded = self.region_embedding(region)
        
        # Иерархические модели
        sku_features = self.sku_model(sku_embedded)
        region_features = self.region_model(region_embedded)
        
        # Объединение признаков
        combined = torch.cat([sku_features, region_features, sales_history.unsqueeze(-1)], dim=-1)
        
        # Трансформер
        transformer_out = self.transformer(combined)
        
        # Внимание
        attn_out, _ = self.attention(transformer_out, transformer_out, transformer_out)
        
        # Тренд
        trend = self.trend_layer(attn_out[:, -1, :])
        
        # Сезонность
        seasonality = self.seasonality_layer(attn_out[:, -1, :])
        
        # Доверительные интервалы
        confidence = self.confidence_layer(attn_out[:, -1, :])
        
        # Объединение всех признаков
        combined_features = torch.cat([attn_out[:, -1, :], trend, seasonality], dim=-1)
        
        # Полносвязные слои
        output = self.fc(combined_features)
        
        return output, trend, seasonality, confidence

