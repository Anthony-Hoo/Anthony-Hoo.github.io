import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/",

  locales: {
    "/": {
      lang: "zh-CN",
      title: "山间之风",
      description: "Hoo的博客",
    },
    "/en/": {
      lang: "en-US",
      title: "山间之风",
      description: "Hoo's blog",
    },
  },

  head: [["link", { rel: "icon", href: "/favicon.webp" }]],

  theme,

  // Enable it with pwa
  // shouldPrefetch: false,
});
