<template>
  <div class="custom-image-container">
    <img src="../assets/images/my-image.jpg" alt="我的图片描述" class="custom-image">
  </div>
</template>

<script>
export default {
  name: 'MyImage'
}
</script>

<style scoped>
.custom-image-container {
  margin: 20px auto;
  text-align: center;
  max-width: 400px;
}

.custom-image {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
</style>