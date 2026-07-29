# Build context is webapp/frontend (see docker-compose.yml) — this one's
# self-contained, unlike the backend, since the frontend has no cross-directory
# imports into the rest of the repo.
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
