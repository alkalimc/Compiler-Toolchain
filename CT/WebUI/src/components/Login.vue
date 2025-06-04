<template>
  <div class="login-container">
    <el-card class="login-card">
      <div slot="header" class="header">
        <span>FPGA Compiler Toolchain</span>
      </div>
      <el-form
          :model="loginForm"
          :rules="rules"
          ref="loginForm"
          @submit.native.prevent="handleLogin">
        <el-form-item label="用户名" prop="username">
          <el-input
              v-model="loginForm.username"
              placeholder="请输入用户名"
              prefix-icon="el-icon-user">
          </el-input>
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input
              v-model="loginForm.password"
              type="password"
              placeholder="请输入密码"
              prefix-icon="el-icon-lock"
              show-password>
          </el-input>
        </el-form-item>
        <el-form-item>
          <el-button
              type="primary"
              native-type="submit"
              :loading="loading"
              class="login-button">
            登录
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'Login',
  data() {
    return {
      loading: false,
      loginForm: {
        username: '',
        password: ''
      },
      rules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' },
          { min: 3, max: 20, message: '长度在3到20个字符', trigger: 'blur' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' }
        ]
      }
    }
  },
  methods: {
    handleLogin() {
      this.$refs.loginForm.validate(valid => {
        if (valid) {
          this.loading = true
          this.verifyCredentials()
        }
      })
    },
    async verifyCredentials() {
      try {
        const authHeader = 'Basic ' + btoa(`${this.loginForm.username}:${this.loginForm.password}`)
        const response = await axios.get('http://10.20.108.87:7678/api/verify', {
          headers: { 'Authorization': authHeader }
        })

        if (response.data.success) {
          this.$emit('login-success', {
            username: this.loginForm.username,
            password: this.loginForm.password
          })
        } else {
          throw new Error(response.data.message || '认证失败')
        }
      } catch (error) {
        // 修改点：增强错误提示
        let errorMsg = '登录失败: '
        if (error.response) {
          errorMsg += error.response.data?.message || error.response.statusText
        } else {
          errorMsg += error.message
        }
        this.$message.error(errorMsg)
        console.error('登录错误详情:', error)
      } finally {
        this.loading = false
      }
    }
  }
}
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background: rgba(255, 255, 255, 0.95);
}

.login-card {
  width: 400px;
  padding: 0;
  border-radius: 8px;
}

.el-form {
  margin-top: 20px;
}

.header {
  text-align: center;
  font-size: 18px;
  font-weight: bold;
}

.login-button {
  width: 100%;
}
</style>