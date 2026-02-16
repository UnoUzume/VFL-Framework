

**学术加速**
source /etc/network_turbo

**安装 UV**
curl -LsSf https://astral.sh/uv/install.sh | sh
*重启*

**在 Jupyter 工作目录创建 VSCode 文件夹**

mkdir VSCode
cd VSCode
mkdir VFL-Framework
cd VFL-Framework
unzip VFL-Framework.zip

`~/VSCode/VFL-Framework#`

uv python install 3.12（uv sync 自动）
uv sync

mkdir -p /root/autodl-tmp/VFL-Framework/data
ln -s /root/autodl-tmp/VFL-Framework/data data



**准备数据集**
mkdir -p data/datasets/cifar10
cp /root/autodl-pub/cifar-10/cifar-10-python.tar.gz data/datasets/cifar10

mkdir -p data/datasets/cinic10/CINIC-10
cp /root/autodl-fs/CINIC-10.tar.gz data/datasets/cinic10

mkdir -p data/datasets/imagenette
cp /root/autodl-fs/imagenette2-160.tgz data/datasets/imagenette

**激活环境**
source .venv/bin/activate

**运行实验**
python -m projects.XXX.XXX

**TensorBoard**
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
tensorboard --port 6007 --logdir data/logs/XXX/lightning_logs

**打包文件**
`zip -r lightning.zip . -x "data/*" ".venv/*" "*.zip"`

**Git**
git config --global user.name "Sasara"
git config --global user.email "sasara@users.noreply.github.com"
