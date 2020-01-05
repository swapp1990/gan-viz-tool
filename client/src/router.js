import Vue from 'vue';
import Router from 'vue-router';

import GANTool from './views/GANTool.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      component: GANTool
    }
  ],
});