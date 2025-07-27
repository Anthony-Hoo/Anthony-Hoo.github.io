import { navbar } from "vuepress-theme-hope";

const rssNavItem = process.env.NODE_ENV === 'production' ? [{
  text: "RSS订阅",
  icon: "rss",
  link: "/rss.xml",
  target: "_blank",
}] : [];

export const zhNavbar = navbar([


]);
