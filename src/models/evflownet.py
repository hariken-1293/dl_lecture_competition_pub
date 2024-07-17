import torch
from torch import nn
from src.models.base import *
from typing import Dict, Any

_BASE_CHANNELS = 64

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet,self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
#         print(f"Inside forward: {inputs.keys()}")  # Debugging line
        event_volume_1 = inputs['event_volume_1']
        event_volume_2 = inputs['event_volume_2']
        combined_input = torch.cat([event_volume_1, event_volume_2], dim=1)
#         print(f"Combined input shape: {combined_input.shape}")  # Debugging line

        try:
            # encoder
            skip_connections = {}
            inputs = self.encoder1(combined_input)  # 修正: combined_inputを渡す
            skip_connections['skip0'] = inputs.clone()
#             print("Passed encoder1")  # Debugging line
            inputs = self.encoder2(inputs)
            skip_connections['skip1'] = inputs.clone()
#             print("Passed encoder2")  # Debugging line
            inputs = self.encoder3(inputs)
            skip_connections['skip2'] = inputs.clone()
#             print("Passed encoder3")  # Debugging line
            inputs = self.encoder4(inputs)
            skip_connections['skip3'] = inputs.clone()
#             print("Passed encoder4")  # Debugging line

            # transition
            inputs = self.resnet_block(inputs)
#             print("Passed resnet_block")  # Debugging line

            # decoder
            flow_dict = {}
            inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
            inputs, flow = self.decoder1(inputs)
            flow_dict['flow0'] = flow.clone()
#             print("Passed decoder1")  # Debugging line

            inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
            inputs, flow = self.decoder2(inputs)
            flow_dict['flow1'] = flow.clone()
#             print("Passed decoder2")  # Debugging line

            inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
            inputs, flow = self.decoder3(inputs)
            flow_dict['flow2'] = flow.clone()
#             print("Passed decoder3")  # Debugging line

            inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
            inputs, flow = self.decoder4(inputs)
            flow_dict['flow3'] = flow.clone()
#             print("Passed decoder4")  # Debugging line

        except Exception as e:
            print(f"Error in forward pass: {e}")

        return flow

        

# if __name__ == "__main__":
#     from config import configs
#     import time
#     from data_loader import EventData
#     '''
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     input_ = torch.rand(8,4,256,256).cuda()
#     a = time.time()
#     output = model(input_)
#     b = time.time()
#     print(b-a)
#     print(output['flow0'].shape, output['flow1'].shape, output['flow2'].shape, output['flow3'].shape)
#     #print(model.state_dict().keys())
#     #print(model)
#     '''
#     import numpy as np
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     EventDataset = EventData(args.data_path, 'train')
#     EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)
#     #model = nn.DataParallel(model)
#     #model.load_state_dict(torch.load(args.load_path+'/model18'))
#     for input_, _, _, _ in EventDataLoader:
#         input_ = input_.cuda()
#         a = time.time()
#         (model(input_))
#         b = time.time()
#         print(b-a)