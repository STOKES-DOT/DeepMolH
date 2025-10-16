import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.trial import TrialState
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 假设你已经有了这些模块
from Hamiltonian_block import Hamiltonian_diagonal_block, Hamiltonian_off_diagonal_block
from NeuralNetwork import DeepMolH
import overlap_block
import bond_embedding
import atom_embedding
import nodes_embedding
import EGAT_Layer
import GAT_Layer
#NOTE: This is a training test code made by DeepSeek R1 !!!
# 设置样式
plt.style.use('default')
sns.set_palette("husl")

class CompleteDeepMolHTrainer:
    def __init__(self, data_dir, use_optuna=True):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_optuna = use_optuna
        self.study = None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.test_predictions = []
        
    def load_dataset(self, train_ratio=0.2, val_ratio=0.1):
        """加载并划分数据集"""
        mol_dir = os.path.join(self.data_dir, 'mol')
        hamiltonian_dir = os.path.join(self.data_dir, 'Hamiltonian')
        
        # 获取所有分子文件
        mol_files = sorted([f for f in os.listdir(mol_dir) if f.endswith('.mol2')])
        
        # 计算划分点
        n_total = len(mol_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # 划分数据集
        train_mol_files = mol_files[:n_train]
        val_mol_files = mol_files[n_train:n_train+n_val]
        test_mol_files = mol_files[n_train+n_val:]
        
        print(f"Dataset split: {len(train_mol_files)} train, {len(val_mol_files)} val, {len(test_mol_files)} test")
        
        train_data, val_data, test_data = [], [], []
        
        # 加载数据
        for split_name, mol_files, data_list in [
            ("Training", train_mol_files, train_data),
            ("Validation", val_mol_files, val_data),
            ("Test", test_mol_files, test_data)
        ]:
            print(f"Loading {split_name} data...")
            for mol_file in tqdm(mol_files):
                mol_path = os.path.join(mol_dir, mol_file)
                mol_id = mol_file.split('.')[0]
                h_path = os.path.join(hamiltonian_dir, f"{mol_id}_h.npz")
                
                if os.path.exists(h_path):
                    H_target = sp.load_npz(h_path).toarray()
                    data_list.append({
                        'mol_id': mol_id,
                        'mol_path': mol_path,
                        'H_target': torch.tensor(H_target, dtype=torch.float32)
                    })
        
        return train_data, val_data, test_data
    
    def create_model(self, trial=None):
        """创建模型，使用Optuna或默认参数"""
        if trial is not None and self.use_optuna:
            # 使用Optuna建议的超参数
            num_egat_layers = trial.suggest_int('num_egat_layers', 1, 20)
            num_gat_layers = trial.suggest_int('num_gat_layers', 1, 20)
            num_heads = trial.suggest_int('num_heads', 1, 16)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            
            model = DeepMolH(
                num_egat_layers=num_egat_layers,
                num_gat_layers=num_gat_layers,
                num_heads=num_heads,
                dropout=dropout
            ).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            return model, optimizer, {
                'num_egat_layers': num_egat_layers,
                'num_gat_layers': num_gat_layers,
                'num_heads': num_heads,
                'dropout': dropout,
                'learning_rate': learning_rate
            }
        else:
            # 使用默认参数
            model = DeepMolH().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            return model, optimizer, {}
    
    def train_epoch(self, model, optimizer, train_data):
        """训练一个epoch"""
        model.train()
        epoch_loss = 0.0
        criterion = nn.MSELoss()
        
        for data in tqdm(train_data, desc="Training", leave=False):
            mol_path = data['mol_path']
            H_target = data['H_target'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            H_pred = model(mol_path)
            
            # 确保矩阵大小一致
            min_size = min(H_pred.shape[0], H_target.shape[0])
            H_pred = H_pred[:min_size, :min_size]
            H_target = H_target[:min_size, :min_size]
            
            # 计算损失
            loss = criterion(H_pred, H_target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_data)
    
    def evaluate(self, model, data):
        """在数据集上评估模型"""
        model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data_point in tqdm(data, desc="Evaluating", leave=False):
                mol_path = data_point['mol_path']
                H_target = data_point['H_target'].to(self.device)
                
                H_pred = model(mol_path)
                
                # 确保矩阵大小一致
                min_size = min(H_pred.shape[0], H_target.shape[0])
                H_pred = H_pred[:min_size, :min_size]
                H_target = H_target[:min_size, :min_size]
                
                loss = criterion(H_pred, H_target)
                total_loss += loss.item()
        
        return total_loss / len(data)
    
    def optuna_objective(self, trial):
        """Optuna目标函数"""
        # 创建模型和优化器
        model, optimizer, hyperparams = self.create_model(trial)
        
        # 加载数据
        train_data, val_data, _ = self.load_dataset()
        
        print(f"\nTrial {trial.number}: {hyperparams}")
        
        # 训练模型
        best_val_loss = float('inf')
        patience = 8
        patience_counter = 0
        
        for epoch in range(50):  # 最大100个epoch用于超参数搜索
            # 训练
            train_loss = self.train_epoch(model, optimizer, train_data)
            
            # 验证
            val_loss = self.evaluate(model, val_data)
            
            # 报告给Optuna
            trial.report(val_loss, epoch)
            
            # 早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
            
            # 检查是否需要剪枝
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_loss
    
    def run_hyperparameter_optimization(self, n_trials=10):
        """运行超参数优化"""
        if not self.use_optuna:
            print("Optuna optimization disabled. Using default parameters.")
            return None
            
        print("Starting hyperparameter optimization with Optuna...")
        
        # 创建研究
        self.study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.HyperbandPruner(),
            study_name="deepmolh_hyperopt"
        )
        
        # 添加回调函数来打印最佳试验
        def print_best_trial(study, trial):
            if study.best_trial.number == trial.number:
                print(f"\n🎯 New best trial {trial.number} with value: {trial.value:.6f}")
                print(f"Best params: {trial.params}")
        
        # 运行优化
        self.study.optimize(self.optuna_objective, n_trials=n_trials, callbacks=[print_best_trial])
        
        # 保存研究
        joblib.dump(self.study, "deepmolh_optuna_study.pkl")
        
        # 生成报告
        self.generate_optuna_report()
        
        return self.study.best_params
    
    def generate_optuna_report(self):
        """生成Optuna优化报告"""
        if self.study is None:
            return
            
        print("\n" + "="*60)
        print("OPTUNA HYPERPARAMETER OPTIMIZATION REPORT")
        print("="*60)
        
        # 最佳试验信息
        best_trial = self.study.best_trial
        print(f"\n🏆 BEST TRIAL (#{best_trial.number})")
        print(f"Validation Loss: {best_trial.value:.6f}")
        print("\nBest Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        # 试验统计
        completed_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        
        print(f"\n📊 TRIAL STATISTICS")
        print(f"Completed trials: {len(completed_trials)}")
        print(f"Pruned trials: {len(pruned_trials)}")
        print(f"Total trials: {len(self.study.trials)}")
        
        # 可视化
        self.plot_optuna_results()
    
    def plot_optuna_results(self):
        """绘制Optuna优化结果"""
        if self.study is None:
            return
            
        try:
            # 优化历史
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.write_image("optimization_history.png")
            
            # 参数重要性
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.write_image("param_importance.png")
            
            print("📈 Optuna visualizations saved as PNG files")
        except Exception as e:
            print(f"⚠️ Could not generate Optuna visualizations: {e}")
    
    def train_final_model(self, best_params=None, num_epochs=100):
        """使用最佳超参数训练最终模型"""
        print("\n" + "="*60)
        print("TRAINING FINAL MODEL")
        print("="*60)
        
        # 创建模型
        if best_params and self.use_optuna:
            print(f"Using optimized hyperparameters: {best_params}")
            model, optimizer, _ = self.create_model(None)
            # 手动设置优化器学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = best_params.get('learning_rate', 0.001)
        else:
            print("Using default hyperparameters")
            model, optimizer, _ = self.create_model(None)
        
        # 加载数据
        train_data, val_data, test_data = self.load_dataset()
        
        criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        print("Starting final training...")
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(model, optimizer, train_data)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.evaluate(model, val_data)
            self.val_losses.append(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'hyperparameters': best_params if best_params else {}
                }, 'best_deepmolh_model.pth')
                print(f"✅ New best model saved at epoch {epoch+1}")
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 早停
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        checkpoint = torch.load('best_deepmolh_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\nFinal Training Results:")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final training loss: {self.train_losses[-1]:.6f}")
        
        # 绘制训练曲线
        self.plot_training_curve()
        
        # 在测试集上评估
        self.test_predictions = self.predict_test_set(model, test_data)
        
        # 可视化测试结果
        self.visualize_test_results()
        
        return model, best_val_loss
    
    def predict_test_set(self, model, test_data):
        """在测试集上进行预测"""
        print("\nPredicting on test set...")
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for data in tqdm(test_data, desc="Predicting"):
                mol_path = data['mol_path']
                H_target = data['H_target'].cpu().numpy()
                
                H_pred = model(mol_path).cpu().numpy()
                
                # 确保矩阵大小一致
                min_size = min(H_pred.shape[0], H_target.shape[0])
                H_pred = H_pred[:min_size, :min_size]
                H_target = H_target[:min_size, :min_size]
                
                predictions.append({
                    'mol_id': data['mol_id'],
                    'mol_path': data['mol_path'],
                    'H_target': H_target,
                    'H_pred': H_pred,
                    'H_diff': np.abs(H_pred - H_target),
                    'mse': np.mean((H_pred - H_target)**2),
                    'mae': np.mean(np.abs(H_pred - H_target)),
                    'max_diff': np.max(np.abs(H_pred - H_target))
                })
        
        return predictions
    
    def visualize_test_results(self, max_display_size=40):
        """可视化测试集结果"""
        if not self.test_predictions:
            print("No test predictions to visualize")
            return
            
        print("\n" + "="*60)
        print("VISUALIZING TEST RESULTS")
        print("="*60)
        
        # 为每个测试分子创建可视化
        for i, pred in enumerate(self.test_predictions):
            mol_id = pred['mol_id']
            H_target = pred['H_target']
            H_pred = pred['H_pred']
            H_diff = pred['H_diff']
            mse = pred['mse']
            
            print(f"\nVisualizing molecule {mol_id} (MSE: {mse:.6f})")
            
            # 限制显示大小
            display_size = min(H_target.shape[0], max_display_size)
            H_target_display = H_target[:display_size, :display_size]
            H_pred_display = H_pred[:display_size, :display_size]
            H_diff_display = H_diff[:display_size, :display_size]
            
            # 创建图形
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Molecule {mol_id} - Hamiltonian Comparison (MSE: {mse:.6f})', fontsize=16, y=0.95)
            
            # 第一行：热图比较
            vmin = min(H_target_display.min(), H_pred_display.min())
            vmax = max(H_target_display.max(), H_pred_display.max())
            
            # 目标哈密顿量
            im1 = axes[0, 0].imshow(H_target_display, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0, 0].set_title('Target Hamiltonian', fontsize=12)
            axes[0, 0].set_xlabel('Orbital Index')
            axes[0, 0].set_ylabel('Orbital Index')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # 预测哈密顿量
            im2 = axes[0, 1].imshow(H_pred_display, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0, 1].set_title('Predicted Hamiltonian', fontsize=12)
            axes[0, 1].set_xlabel('Orbital Index')
            axes[0, 1].set_ylabel('Orbital Index')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # 差异热图
            im3 = axes[0, 2].imshow(H_diff_display, cmap='hot', aspect='auto')
            axes[0, 2].set_title('Absolute Difference', fontsize=12)
            axes[0, 2].set_xlabel('Orbital Index')
            axes[0, 2].set_ylabel('Orbital Index')
            plt.colorbar(im3, ax=axes[0, 2])
            
            # 第二行：详细分析
            # 对角元素比较
            diag_target = np.diag(H_target_display)
            diag_pred = np.diag(H_pred_display)
            axes[1, 0].plot(diag_target, 'bo-', label='Target', alpha=0.7, markersize=3)
            axes[1, 0].plot(diag_pred, 'ro-', label='Predicted', alpha=0.7, markersize=3)
            axes[1, 0].set_title('Diagonal Elements Comparison', fontsize=12)
            axes[1, 0].set_xlabel('Orbital Index')
            axes[1, 0].set_ylabel('Hamiltonian Element Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 对角元素差异
            diag_diff = np.abs(diag_target - diag_pred)
            axes[1, 1].bar(range(len(diag_diff)), diag_diff, alpha=0.7, color='purple')
            axes[1, 1].set_title('Diagonal Elements Absolute Difference', fontsize=12)
            axes[1, 1].set_xlabel('Orbital Index')
            axes[1, 1].set_ylabel('Absolute Difference')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 非对角元素散点图
            off_diag_mask = ~np.eye(display_size, dtype=bool)
            off_diag_target = H_target_display[off_diag_mask]
            off_diag_pred = H_pred_display[off_diag_mask]
            
            # 随机采样以避免过多点
            if len(off_diag_target) > 1000:
                indices = np.random.choice(len(off_diag_target), 1000, replace=False)
                off_diag_target = off_diag_target[indices]
                off_diag_pred = off_diag_pred[indices]
            
            axes[1, 2].scatter(off_diag_target, off_diag_pred, alpha=0.5, s=2, color='green')
            min_val = min(off_diag_target.min(), off_diag_pred.min())
            max_val = max(off_diag_target.max(), off_diag_pred.max())
            axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            axes[1, 2].set_title('Off-diagonal: Target vs Predicted', fontsize=12)
            axes[1, 2].set_xlabel('Target Values')
            axes[1, 2].set_ylabel('Predicted Values')
            axes[1, 2].grid(True, alpha=0.3)
            
            # 计算相关系数
            if len(off_diag_target) > 1:
                correlation = np.corrcoef(off_diag_target, off_diag_pred)[0, 1]
                axes[1, 2].text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                               transform=axes[1, 2].transAxes, 
                               bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'test_result_{mol_id}.png', dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形以避免显示
            
            # 打印统计信息
            print(f"  Matrix size: {H_target.shape[0]}x{H_target.shape[1]}")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {pred['mae']:.6f}")
            print(f"  Max absolute difference: {pred['max_diff']:.6f}")
            if len(off_diag_target) > 1:
                print(f"  Off-diagonal correlation: {correlation:.6f}")
        
        # 创建测试集总结报告
        self.create_test_summary_report()
    
    def create_test_summary_report(self):
        """创建测试集总结报告"""
        if not self.test_predictions:
            return
            
        print("\n" + "="*60)
        print("TEST SET SUMMARY REPORT")
        print("="*60)
        
        summary_data = []
        for pred in self.test_predictions:
            mol_id = pred['mol_id']
            H_target = pred['H_target']
            
            # 计算对角元素统计
            diag_target = np.diag(H_target)
            diag_pred = np.diag(pred['H_pred'])
            diag_mse = np.mean((diag_target - diag_pred)**2)
            
            # 非对角元素统计
            off_diag_mask = ~np.eye(H_target.shape[0], dtype=bool)
            off_diag_target = H_target[off_diag_mask]
            off_diag_pred = pred['H_pred'][off_diag_mask]
            
            if len(off_diag_target) > 1:
                off_diag_corr = np.corrcoef(off_diag_target, off_diag_pred)[0, 1]
            else:
                off_diag_corr = 0
            
            summary_data.append({
                'Molecule': mol_id,
                'Matrix_Size': f"{H_target.shape[0]}x{H_target.shape[1]}",
                'MSE': pred['mse'],
                'MAE': pred['mae'],
                'Max_Diff': pred['max_diff'],
                'Diag_MSE': diag_mse,
                'OffDiag_Corr': off_diag_corr
            })
            
            print(f"\n{mol_id}:")
            print(f"  Size: {H_target.shape[0]}x{H_target.shape[1]}")
            print(f"  MSE: {pred['mse']:.6f}")
            print(f"  MAE: {pred['mae']:.6f}")
            print(f"  Max Diff: {pred['max_diff']:.6f}")
            print(f"  Diagonal MSE: {diag_mse:.6f}")
            print(f"  Off-diagonal Correlation: {off_diag_corr:.6f}")
        
        # 计算平均值
        if summary_data:
            avg_mse = np.mean([d['MSE'] for d in summary_data])
            avg_mae = np.mean([d['MAE'] for d in summary_data])
            avg_diag_mse = np.mean([d['Diag_MSE'] for d in summary_data])
            avg_offdiag_corr = np.mean([d['OffDiag_Corr'] for d in summary_data])
            
            print(f"\n=== AVERAGES ===")
            print(f"Average MSE: {avg_mse:.6f}")
            print(f"Average MAE: {avg_mae:.6f}")
            print(f"Average Diagonal MSE: {avg_diag_mse:.6f}")
            print(f"Average Off-diagonal Correlation: {avg_offdiag_corr:.6f}")
            
            # 保存总结报告
            df = pd.DataFrame(summary_data)
            df.to_csv('test_set_summary.csv', index=False)
            print(f"\n💾 Test set summary saved to 'test_set_summary.csv'")
        
        return summary_data
    
    def plot_training_curve(self):
        """绘制训练曲线"""
        if not self.train_losses or not self.val_losses:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DeepMolH Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_pipeline(self, n_trials=20, num_epochs=100):
        """运行完整的训练管道"""
        print("🚀 Starting Complete DeepMolH Training Pipeline")
        print("="*60)
        
        # 1. 超参数优化（如果启用）
        best_params = None
        if self.use_optuna:
            best_params = self.run_hyperparameter_optimization(n_trials=n_trials)
        
        # 2. 训练最终模型
        model, final_val_loss = self.train_final_model(best_params, num_epochs=num_epochs)
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final validation loss: {final_val_loss:.6f}")
        print(f"Test set predictions: {len(self.test_predictions)} molecules")
        print("\nGenerated files:")
        print("  - best_deepmolh_model.pth (trained model)")
        print("  - training_curve.png (training progress)")
        print("  - test_result_*.png (test molecule visualizations)")
        print("  - test_set_summary.csv (test results summary)")
        if self.use_optuna:
            print("  - deepmolh_optuna_study.pkl (Optuna study)")
            print("  - optimization_history.png (Optuna optimization history)")
            print("  - param_importance.png (hyperparameter importance)")
        
        return model, final_val_loss

def main():
    # 配置
    data_dir = '/Users/jiaoyuan/Documents/GitHub/DeepMolH/DeepMolH/dataset'
    use_optuna = True  # 设置为False跳过超参数优化
    n_trials = 20      # Optuna试验次数
    num_epochs = 100   # 最终训练轮数
    
    # 创建训练器
    trainer = CompleteDeepMolHTrainer(data_dir, use_optuna=use_optuna)
    
    # 运行完整管道
    model, final_loss = trainer.run_complete_pipeline(
        n_trials=n_trials, 
        num_epochs=num_epochs
    )
    
    return model, trainer

if __name__ == '__main__':
    model, trainer = main()