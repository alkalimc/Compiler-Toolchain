<template>
  <el-card>
    <div slot="header" class="header-with-icon">
      <img src="../assets/header-icon.jpg" class="header-icon">
      <span>大模型 FPGA 部署工具</span>
    </div>

    <el-form label-position="top">
      <!-- 模型选择模块 -->
      <el-form-item label="1. 选择模型">
        <el-select v-model="selectedModel" placeholder="请选择模型">
          <el-option
              v-for="model in models"
              :key="model.value"
              :label="model.label"
              :value="model.value">
            <span style="float: left">{{ model.label }}</span>
            <img :src="model.icon" class="option-icon">
          </el-option>
        </el-select>
      </el-form-item>

      <!-- 量化方式模块 -->
      <el-form-item label="2. 选择量化方式">
        <el-radio-group v-model="selectedQuant">
          <el-radio v-for="quant in quants"
                    :label="quant.value"
                    :key="quant.value">
            <img :src="quant.icon" class="radio-icon">
            {{ quant.label }}
          </el-radio>
        </el-radio-group>
      </el-form-item>

      <!-- FPGA 部署模块 -->
      <el-form-item class="deploy-section">
        <div class="deploy-button-wrapper">
          <el-button type="primary" @click="startDeploy" :loading="isDeploying">
            <img src="../assets/deploy-icon.jpg" class="button-icon">
            开始部署
          </el-button>
        </div>

        <el-alert v-if="deployStatus.length > 0"
                  :title="''"
                  type="info"
                  :closable="false"
                  :show-icon="false"
                  class="status-alert">
          <div>
            <div v-for="(line, index) in deployStatus" :key="index">
              {{ line }}
            </div>
          </div>
        </el-alert>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<script>
import modelQwen from '../assets/model-qwen.jpg'
import modelGpt from '../assets/model-gpt.jpg'
import quantGptq from '../assets/quant-gptq.jpg'
import quantAwq from '../assets/quant-awq.jpg'
import quantMixq from '../assets/quant-mixq.jpg'

export default {
  name: 'DeploymentTool',
  data() {
    return {
      selectedModel: '',
      selectedQuant: 'gptq',
      isDeploying: false,
      deployStatus: [],
      models: [
        { value: 'qwen2', label: 'Qwen2-7B', icon: modelQwen, size: 15 },
        { value: 'qwen2.5', label: 'Qwen2.5-72B', icon: modelQwen, size: 140 },
        { value: 'gpt4', label: 'GPT-4', icon: modelGpt, size: 200 }
      ],
      quants: [
        { value: 'gptq', label: 'GPTQ', icon: quantGptq },
        { value: 'awq', label: 'AWQ', icon: quantAwq },
        { value: 'mixq', label: 'MixQ', icon: quantMixq }
      ]
    }
  },
  methods: {
    async startDeploy() {
      if (!this.selectedModel) {
        this.$message.error('请先选择模型');
        return;
      }

      this.isDeploying = true;
      this.deployStatus = [];

      try {
        // 1. 下载模型
        await this.downloadModel();

        // 2. 量化处理
        this.deployStatus.push(`2. 量化中 (${this.getQuantName(this.selectedQuant)})...`);
        await this.delay(1500);
        this.deployStatus.push('量化完成');

        // 3. 生成FPGA代码
        this.deployStatus.push('3. 生成 FPGA 代码...');
        await this.delay(1500);
        this.deployStatus.push('FPGA 代码生成完成');

        // 4. 完成
        this.deployStatus.push('✅ 部署成功！');
        this.$emit('deploy-success', {
          name: this.getModelName(this.selectedModel),
          quant: this.getQuantName(this.selectedQuant)
        })
      } catch (error) {
        this.deployStatus.push(`❌ 部署失败: ${error.message}`);
      } finally {
        this.isDeploying = false;
      }
    },

    async downloadModel() {
      const model = this.getCurrentModel();
      this.deployStatus.push(`1. 开始下载模型: ${model.label} (${model.size}GB)...`);
      await this.delay(5000);
      this.deployStatus.push('下载完成');
    },

    delay(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    },

    getCurrentModel() {
      return this.models.find(m => m.value === this.selectedModel);
    },

    getModelName(value) {
      const model = this.models.find(m => m.value === value);
      return model ? model.label : '';
    },

    getQuantName(value) {
      const quant = this.quants.find(q => q.value === value);
      return quant ? quant.label : '';
    }
  }
}
</script>

<style scoped>

.header-with-icon {
  display: flex;
  align-items: center;
  justify-content: center; /* 添加水平居中 */
  font-size: 20px;
  font-weight: bold;
  width: 100%; /* 确保宽度填满 */
  text-align: center; /* 作为备用方案 */
}

.header-icon {
  width: 32px;
  height: 32px;
  margin-right: 10px;
}

.option-icon {
  width: 20px;
  height: 20px;
  float: right;
  margin-top: 2px;
}

.radio-icon {
  width: 16px;
  height: 16px;
  vertical-align: middle;
  margin-right: 5px;
}

.button-icon {
  width: 18px;
  height: 18px;
  vertical-align: middle;
  margin-right: 5px;
}

.el-card {
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

::v-deep .el-form-item__label {
  font-size: 16px;
  font-weight: bold;
  color: #333;
}

/* 修改后的样式 */
.deploy-section {
  margin-top: 20px;
  display: flex;
  flex-direction: column;
  align-items: center; /* 使子元素水平居中 */
}

.deploy-button-wrapper {
  width: 100%; /* 确保容器宽度足够 */
  display: flex;
  justify-content: center;
  margin-bottom: 20px;
}

.status-alert {
  margin-top: 10px;
  width: 100%; /* 确保状态框宽度与表单一致 */
}

/* 修复 el-form-item 的默认样式影响 */
.el-form-item__content {
  display: flex;
  flex-direction: column;
  align-items: center;
}
</style>