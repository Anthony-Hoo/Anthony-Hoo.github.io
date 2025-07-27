import { navbar } from "vuepress-theme-hope";

const rssNavItem = process.env.NODE_ENV === 'production' ? [{
  text: "RSS Feed",
  icon: "rss",
  link: "/en/rss.xml",
  target: "_blank",
}] : [];

export const enNavbar = navbar([
  "/en/",
  {
    text: "Posts",
    icon: "pen-to-square",
    prefix: "/en/posts/",
    children: [
      { text: "Hello World", icon: "pen-to-square", link: "helloworld" },
    ],
  },


]);
