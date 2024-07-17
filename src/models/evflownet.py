import torch
from torch import nn
from src.models.base import *
from typing import Dict, Any

_BASE_CHANNELS = 64


class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet, self).__init__()
        self._args = args
        self.dropout = nn.Dropout(p=0.5)  # Dropout層を追加

        self.encoder1 = general_conv2d(
            in_channels=4,
            out_channels=_BASE_CHANNELS,
            do_batch_norm=not self._args.no_batch_norm,
        )
        self.encoder2 = general_conv2d(
            in_channels=_BASE_CHANNELS,
            out_channels=2 * _BASE_CHANNELS,
            do_batch_norm=not self._args.no_batch_norm,
        )
        self.encoder3 = general_conv2d(
            in_channels=2 * _BASE_CHANNELS,
            out_channels=4 * _BASE_CHANNELS,
            do_batch_norm=not self._args.no_batch_norm,
        )
        self.encoder4 = general_conv2d(
            in_channels=4 * _BASE_CHANNELS,
            out_channels=8 * _BASE_CHANNELS,
            do_batch_norm=not self._args.no_batch_norm,
        )

        self.resnet_block = nn.Sequential(
            *[
                build_resnet_block(
                    8 * _BASE_CHANNELS,
                    do_batch_norm=not self._args.no_batch_norm,
                )
                for i in range(2)
            ]
        )

        self.decoder1 = upsample_conv2d_and_predict_flow(
            in_channels=16 * _BASE_CHANNELS,
            out_channels=4 * _BASE_CHANNELS,
            do_batch_norm=not self._args.no_batch_norm,
        )

        self.decoder2 = upsample_conv2d_and_predict_flow(
            in_channels=8 * _BASE_CHANNELS + 2,
            out_channels=2 * _BASE_CHANNELS,
            do_batch_norm=not self._args.no_batch_norm,
        )

        self.decoder3 = upsample_conv2d_and_predict_flow(
            in_channels=4 * _BASE_CHANNELS + 2,
            out_channels=_BASE_CHANNELS,
            do_batch_norm=not self._args.no_batch_norm,
        )

        self.decoder4 = upsample_conv2d_and_predict_flow(
            in_channels=2 * _BASE_CHANNELS + 2,
            out_channels=int(_BASE_CHANNELS / 2),
            do_batch_norm=not self._args.no_batch_norm,
        )

    # def compute_loss(
    #     self, pred_flow: torch.Tensor, gt_flow: torch.Tensor
    # ) -> torch.Tensor:
    #     # MSEの計算
    #     mse_loss = torch.mean((pred_flow - gt_flow) ** 2)

    #     # L2正則化項の計算
    #     l2_reg = torch.tensor(0.0, requires_grad=True)
    #     for name, param in self.named_parameters():
    #         if "weight" in name:  # バイアス項は除外
    #             l2_reg = l2_reg + torch.norm(param, 2)

    #     # 総損失の計算（MSE + L2正則化）
    #     total_loss = mse_loss + self.l2_lambda * l2_reg

    #     return total_loss
    def compute_loss(self, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        # MSEの計算
        mse_loss = torch.mean((pred_flow - gt_flow) ** 2)
        return mse_loss

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        event_volume_1 = inputs["event_volume_1"]
        event_volume_2 = inputs["event_volume_2"]
        combined_input = torch.cat([event_volume_1, event_volume_2], dim=1)
        # loss = 0  # ロスの初期化

        try:
            # encoder
            skip_connections = {}
            inputs = self.encoder1(combined_input)
            inputs = self.dropout(inputs)  # Dropoutを適用
            skip_connections["skip0"] = inputs.clone()
            inputs = self.encoder2(inputs)
            inputs = self.dropout(inputs)  # Dropoutを適用
            skip_connections["skip1"] = inputs.clone()
            inputs = self.encoder3(inputs)
            inputs = self.dropout(inputs)  # Dropoutを適用
            skip_connections["skip2"] = inputs.clone()
            inputs = self.encoder4(inputs)
            inputs = self.dropout(inputs)  # Dropoutを適用
            skip_connections["skip3"] = inputs.clone()

            # transition
            inputs = self.resnet_block(inputs)
            inputs = self.dropout(inputs)  # Dropoutを適用

            # decoder and compute losses
            inputs = torch.cat([inputs, skip_connections["skip3"]], dim=1)
            inputs, flow = self.decoder1(inputs)
            # loss += self.compute_loss(inputs, skip_connections["skip2"])

            inputs = torch.cat([inputs, skip_connections["skip2"]], dim=1)
            inputs, flow = self.decoder2(inputs)
            # loss += self.compute_loss(inputs, skip_connections["skip1"])

            inputs = torch.cat([inputs, skip_connections["skip1"]], dim=1)
            inputs, flow = self.decoder3(inputs)
            # loss += self.compute_loss(inputs, skip_connections["skip0"])

            inputs = torch.cat([inputs, skip_connections["skip0"]], dim=1)
            inputs, flow = self.decoder4(inputs)
            # loss += self.compute_loss(inputs, combined_input)  # 最終的な出力との比較

        except Exception as e:
            print(f"Error in forward pass: {e}")
            flow = None

        return flow
