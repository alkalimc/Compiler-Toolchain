<template>
  <el-card class="deployment-tool-card">
    <div slot="header" class="header-with-icon">
      <img src="@/assets/header-icon.jpg" class="header-icon">
      <span>部署模型管理</span>
    </div>

    <el-form label-position="top">
      <!-- 模型名称选择 -->
      <el-form-item label="模型名称">
        <el-select v-model="selectedModel" placeholder="请选择模型">
          <el-option
              v-for="model in models"
              :key="model.value"
              :label="model.label"
              :value="model.value">
            <span style="float: left">{{ model.label }}</span>
            <img :src="model.icon" class="option-icon" />
          </el-option>
        </el-select>
      </el-form-item>

      <!-- 部署类型选择 -->
      <el-form-item label="部署类型">
        <el-select v-model="deployType" placeholder="请选择部署类型">
          <el-option
              v-for="type in deployTypes"
              :key="type.value"
              :label="type.label"
              :value="type.value">
            <span style="float: left">{{ type.label }}</span>
          </el-option>
        </el-select>
      </el-form-item>

      <!-- 启动部署按钮 -->
      <div class="deploy-button-wrapper">
        <el-form-item>
          <el-button type="primary"
                     :loading="isDeploying"
                     :disabled="!selectedModel || !deployType"
                     @click="startDeploy">
            启动部署
          </el-button>
        </el-form-item>
      </div>

      <!-- 部署日志展示区域 -->
      <el-card class="deploy-log-card" v-if="deployLogs.length > 0">
        <div class="log-title">部署日志</div>
        <div class="log-content">
          <div v-for="(log, index) in deployLogs" :key="index" class="log-line">{{ log }}</div>
        </div>
      </el-card>

    </el-form>
  </el-card>
</template>

<script>
import modelQwen from '@/assets/model-qwen.jpg';
import modelDeepseek from '@/assets/model-deepseek.jpg';

export default {
  name: "DeployedModels",
  data() {
    return {
      selectedModel: null,
      deployType: null,
      isDeploying: false,
      deployLogs: [],
      models: [
        { value: 'qwen2', label: 'Qwen2-7B-Instruct', icon: modelQwen },
        { value: 'qwen2.5', label: 'Qwen2.5-7B-Instruct', icon: modelQwen },
        { value: 'qwen2-vl', label: 'Qwen2-VL-7B-Instruct', icon: modelQwen },
        { value: 'qwen2.5-vl', label: 'Qwen2.5-VL-7B-Instruct', icon: modelQwen },
        { value: 'deepseek', label: 'DeepSeek-R1-Distill-Qwen-7B', icon: modelDeepseek }
      ],
      deployTypes: [
        { value: 'original', label: '部署原模型' },
        { value: 'quantized', label: '部署量化模型' },
        { value: 'both', label: '部署两种模型' }
      ]
    };
  }
};
</script>

<style scoped>
.deployment-tool-card {
  width: 500px;
  margin: 0 auto;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.header-with-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  font-weight: bold;
  width: 100%;
  text-align: center;
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

.deploy-button-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 20px;
}

.deploy-log-card {
  margin-top: 20px;
  padding: 10px;
  max-height: 200px;
  overflow-y: auto;
  background-color: #f9f9f9;
  border: 1px solid #ebeef5;
  border-radius: 8px;
}

.log-title {
  font-weight: bold;
  margin-bottom: 10px;
  font-size: 16px;
}

.log-content {
  font-family: monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.log-line {
  margin-bottom: 4px;
}

</style>
