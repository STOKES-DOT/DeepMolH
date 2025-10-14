import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn.o3 as o3
import e3nn.layers as e3nn_layers  # 适配旧版e3nn：用layers.Linear替代nn.Linear


class EquivariantHamiltonianDiagonalBlock(nn.Module):
    def __init__(self, atom_block_size=None, nodes_feature_size=None):
        super().__init__()
        self.nodes_size = nodes_feature_size  # 输入节点特征维度（标量）
        self.orbital_size = atom_block_size   # Hamiltonian对角块维度（如5→5x5矩阵）
        self.num_off_diag_elements = self.orbital_size * (self.orbital_size - 1) // 2  # 上三角非对角元素数

        # -------------------------- 1. 定义各层不可约表示（均为旋转不变标量：nx0e） --------------------------
        # 节点特征输入/输出表示
        self.irreps_nodes_in = o3.Irreps(f"{self.nodes_size}x0e")
        self.irreps_hidden = o3.Irreps(f"{self.orbital_size**2}x0e")  # 隐藏层：orbital_size²个标量
        self.irreps_nodes_out = o3.Irreps(f"{self.orbital_size}x0e")  # 节点特征输出：orbital_size个标量
        # 对角块网络表示
        self.irreps_diag_in = o3.Irreps(f"{self.orbital_size}x0e")
        self.irreps_diag_out = o3.Irreps(f"{self.orbital_size}x0e")
        # 非对角块网络表示
        self.irreps_off_diag_out = o3.Irreps(f"{self.num_off_diag_elements}x0e")

        # -------------------------- 2. 等变节点特征处理网络 --------------------------
        self.nodes_feature_net = nn.Sequential(
            e3nn_layers.Linear(irreps_in=self.irreps_nodes_in, irreps_out=self.irreps_hidden),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_hidden, irreps_out=self.irreps_hidden),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_hidden, irreps_out=self.irreps_hidden),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_hidden, irreps_out=self.irreps_nodes_out),
        )

        # -------------------------- 3. 等变对角块生成网络 --------------------------
        self.H_diag_net_diag = nn.Sequential(
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_out, irreps_out=self.irreps_diag_out),
        )

        # -------------------------- 4. 等变非对角块生成网络 --------------------------
        self.H_diag_net_off_diag = nn.Sequential(
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_diag_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_diag_in, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
        )

        # -------------------------- 5. 物理参数（标量参数不影响等变性） --------------------------
        self.coupling_strength = nn.Parameter(torch.tensor(0.1))
        self.off_diag_coupling_strength = nn.Parameter(torch.tensor(0.1))
        self.edges_contribution = nn.Parameter(torch.tensor(0.1))
        self.orbital_coupling_strength = nn.Parameter(torch.tensor(0.1))
        self.off_diag_orbital_coupling_strength = nn.Parameter(torch.tensor(0.1))

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """统一初始化普通线性层和e3nn等变层的权重"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, e3nn_layers.Linear)):
                nn.init.xavier_uniform_(m.weight)
                # e3nn.layers.Linear可能无bias（默认无），需判断是否存在
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, nodes_feature, edges_feature, atom_block):
        """
        输入：
        - nodes_feature: (nodes_size,) 或 (batch, nodes_size) 节点标量特征
        - edges_feature: (1,) 或 (batch, 1) 边标量特征
        - atom_block: (orbital_size,) 或 (batch, orbital_size) 原子基础标量特征
        输出：
        - H_diag: (1, orbital_size, orbital_size) 或 (batch, orbital_size, orbital_size) 等变Hamiltonian对角块
        """
        # 适配batch维度（默认1个样本）
        if nodes_feature.ndim == 1:
            nodes_feature = nodes_feature.unsqueeze(0)
        if edges_feature.ndim == 1:
            edges_feature = edges_feature.unsqueeze(0)
        if atom_block.ndim == 1:
            atom_block = atom_block.unsqueeze(0)
        device = nodes_feature.device

        # 1. 等变处理节点特征（融合边特征）
        nodes_feature = self.nodes_feature_net(
            nodes_feature + self.edges_contribution * edges_feature
        )

        # 2. 生成对角块（标量→对角矩阵）
        temp_diag_block = self.H_diag_net_diag(nodes_feature)
        diag_block = self.coupling_strength * temp_diag_block
        # 扩展为对角矩阵：(batch, orbital_size, orbital_size)
        diag_block = torch.einsum("bi,ij->bij", diag_block, torch.eye(self.orbital_size, device=device))

        # 3. 融合原子基础块
        base_H = self.orbital_coupling_strength * torch.einsum("bi,ij->bij", atom_block, torch.eye(self.orbital_size, device=device))
        H_diag = diag_block + base_H

        # 4. 生成非对角块（标量→上三角矩阵）
        off_diag_elements = self.H_diag_net_off_diag(nodes_feature)
        off_diag_elements = self.off_diag_coupling_strength * off_diag_elements
        # 初始化上三角矩阵并填充非对角元素
        off_diag_matrix = torch.zeros(H_diag.shape, device=device)
        triu_indices = torch.triu_indices(self.orbital_size, self.orbital_size, offset=1)
        off_diag_matrix[:, triu_indices[0], triu_indices[1]] = off_diag_elements.squeeze(-1)

        # 5. 整合并对称化（能量矩阵需对称）
        H_diag = H_diag + off_diag_matrix
        H_diag = (H_diag + H_diag.transpose(1, 2)) * 0.5

        # 若输入无batch维度，输出也移除batch维度（可选）
        if H_diag.shape[0] == 1:
            H_diag = H_diag.squeeze(0)

        return H_diag


class EquivariantHamiltonianOffDiagonalBlock(nn.Module):
    def __init__(self, atom_in_block_size_rows=None, atom_in_block_size_cols=None, nodes_feature_i_size=None, nodes_feature_j_size=None):
        super().__init__()
        self.nodes_feature_i_size = nodes_feature_i_size
        self.nodes_feature_j_size = nodes_feature_j_size
        self.nodes_size = nodes_feature_i_size  # 节点特征维度（标量，默认i/j维度一致）
        self.rows = atom_in_block_size_rows    # 非对角块行数
        self.cols = atom_in_block_size_cols    # 非对角块列数
        self.orbital_size_rows = self.rows
        self.orbital_size_cols = self.cols

        # -------------------------- 1. 定义不可约表示（均为标量） --------------------------
        # i节点特征表示
        self.irreps_node_i_in = o3.Irreps(f"{self.nodes_size}x0e")
        self.irreps_node_i_out = o3.Irreps(f"{self.orbital_size_cols}x0e")
        # j节点特征表示
        self.irreps_node_j_out = o3.Irreps(f"{self.orbital_size_rows}x0e")
        # 拼接特征表示（i+j节点特征维度之和）
        self.irreps_cat_in = o3.Irreps(f"{self.orbital_size_cols + self.orbital_size_rows}x0e")
        # 非对角块输出表示（行数×列数个标量）
        self.irreps_off_diag_out = o3.Irreps(f"{self.orbital_size_rows * self.orbital_size_cols}x0e")

        # -------------------------- 2. 等变i节点特征处理网络 --------------------------
        self.nodes_feature_net_i = nn.Sequential(
            e3nn_layers.Linear(irreps_in=self.irreps_node_i_in, irreps_out=self.irreps_node_i_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_node_i_out, irreps_out=self.irreps_node_i_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_node_i_out, irreps_out=self.irreps_node_i_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_node_i_out, irreps_out=self.irreps_node_i_out),
        )

        # -------------------------- 3. 等变j节点特征处理网络 --------------------------
        self.nodes_feature_net_j = nn.Sequential(
            e3nn_layers.Linear(irreps_in=self.irreps_node_i_in, irreps_out=self.irreps_node_j_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_node_j_out, irreps_out=self.irreps_node_j_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_node_j_out, irreps_out=self.irreps_node_j_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_node_j_out, irreps_out=self.irreps_node_j_out),
        )

        # -------------------------- 4. 等变非对角块主网络 --------------------------
        self.H_off_diag_net = nn.Sequential(
            e3nn_layers.Linear(irreps_in=self.irreps_cat_in, irreps_out=self.irreps_cat_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_cat_in, irreps_out=self.irreps_cat_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_cat_in, irreps_out=self.irreps_cat_in),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_cat_in, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
            nn.ELU(),
            e3nn_layers.Linear(irreps_in=self.irreps_off_diag_out, irreps_out=self.irreps_off_diag_out),
        )

        # -------------------------- 5. 物理参数 --------------------------
        self.off_diag_coupling_strength = nn.Parameter(torch.tensor(0.1))
        self.nodes_contribution_i = nn.Parameter(torch.tensor(0.1))
        self.nodes_contribution_j = nn.Parameter(torch.tensor(0.1))
        self.edges_contribution_nodes_i = nn.Parameter(torch.tensor(0.1))
        self.edges_contribution_nodes_j = nn.Parameter(torch.tensor(0.1))
        self.orbital_coupling_strength = nn.Parameter(torch.tensor(0.1))

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """统一初始化权重，兼容e3nn.layers.Linear"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, e3nn_layers.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, -0.1)

    def forward(self, nodes_feature_i, nodes_feature_j, edges_feature, atom_in_block):
        """
        输入：
        - nodes_feature_i: (nodes_size,) 或 (batch, nodes_size) i节点标量特征
        - nodes_feature_j: (nodes_size,) 或 (batch, nodes_size) j节点标量特征
        - edges_feature: (1,) 或 (batch, 1) 边标量特征
        - atom_in_block: (rows, cols) 或 (batch, rows, cols) 原子基础矩阵
        输出：
        - H_off_diag: (rows, cols) 或 (batch, rows, cols) 等变Hamiltonian非对角块
        """
        # 适配batch维度（默认1个样本）
        if nodes_feature_i.ndim == 1:
            nodes_feature_i = nodes_feature_i.unsqueeze(0)
        if nodes_feature_j.ndim == 1:
            nodes_feature_j = nodes_feature_j.unsqueeze(0)
        if edges_feature.ndim == 1:
            edges_feature = edges_feature.unsqueeze(0)
        if atom_in_block.ndim == 2:
            atom_in_block = atom_in_block.unsqueeze(0)
        device = nodes_feature_i.device

        # 1. 等变处理i/j节点特征（融合边特征）
        nodes_feature_i = self.nodes_feature_net_i(
            nodes_feature_i + self.edges_contribution_nodes_i * edges_feature
        )
        nodes_feature_j = self.nodes_feature_net_j(
            nodes_feature_j + self.edges_contribution_nodes_j * edges_feature
        )

        # 2. 拼接i/j节点特征（标量拼接，不破坏等变性）
        node_edge_contribution = torch.cat((nodes_feature_i, nodes_feature_j), dim=1)

        # 3. 等变生成非对角块（标量→矩阵）
        H_off_diag = self.H_off_diag_net(node_edge_contribution)
        # 重塑为矩阵：(batch, rows, cols)
        H_off_diag = H_off_diag.view(-1, self.orbital_size_rows, self.orbital_size_cols)

        # 4. 融合原子基础块
        H_off_diag = H_off_diag + self.off_diag_coupling_strength * atom_in_block.to(device)

        # 若输入无batch维度，输出也移除batch维度（可选）
        if H_off_diag.shape[0] == 1:
            H_off_diag = H_off_diag.squeeze(0)

        return H_off_diag


# -------------------------- 测试代码：验证是否运行成功 --------------------------
if __name__ == "__main__":
    # 设备配置（CPU/GPU均可）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 1. 测试对角块
    diag_block = EquivariantHamiltonianDiagonalBlock(
        atom_block_size=5,        # 5x5 Hamiltonian对角块
        nodes_feature_size=16     # 16维节点标量特征
    ).to(device)
    # 生成测试输入
    nodes_feat_diag = torch.randn(16, device=device)
    edges_feat_diag = torch.randn(1, device=device)
    atom_block_diag = torch.randn(5, device=device)
    # 前向传播
    H_diag = diag_block(nodes_feat_diag, edges_feat_diag, atom_block_diag)
    print(f"\n对角块输出形状：{H_diag.shape}")  # 期望：torch.Size([5, 5])
    print(f"对角块对称性验证：{torch.allclose(H_diag, H_diag.T, atol=1e-6)}")  # 期望：True

    # 2. 测试非对角块
    off_diag_block = EquivariantHamiltonianOffDiagonalBlock(
        atom_in_block_size_rows=5,  # 非对角块行数
        atom_in_block_size_cols=4,  # 非对角块列数
        nodes_feature_i_size=16,    # i节点特征维度
        nodes_feature_j_size=16     # j节点特征维度
    ).to(device)
    # 生成测试输入
    nodes_feat_i = torch.randn(16, device=device)
    nodes_feat_j = torch.randn(16, device=device)
    edges_feat_off = torch.randn(1, device=device)
    atom_block_off = torch.randn(5, 4, device=device)
    # 前向传播
    H_off_diag = off_diag_block(nodes_feat_i, nodes_feat_j, edges_feat_off, atom_block_off)
    print(f"\n非对角块输出形状：{H_off_diag.shape}")  # 期望：torch.Size([5, 4])