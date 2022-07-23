
###
 # @Author: JinYin
 # @Date: 2022-07-01 21:43:42
 # @LastEditors: JinYin
 # @LastEditTime: 2022-07-23 14:24:15
 # @FilePath: \CrossDomainFramework\train.sh
 # @Description: 
### 
python train.py --denoise_network 'ResCNN' --mode 'Src' --channel_type "single_channel";
python train.py --denoise_network 'FCNN' --mode 'CD' --channel_type "multi_channel";
python train.py --denoise_network 'FCNN' --mode 'DTD' --channel_type "multi_channel";

