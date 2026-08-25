FROM golang:1.26 AS builder

WORKDIR /src/backend

COPY backend/go.mod backend/go.sum ./
RUN go mod download

COPY backend/ ./
RUN CGO_ENABLED=0 go build -o /out/daybid-dev-service .

FROM alpine:3.22

RUN apk add --no-cache ca-certificates

WORKDIR /app

COPY --from=builder /out/daybid-dev-service /usr/local/bin/daybid-dev-service

EXPOSE 8080

CMD ["daybid-dev-service"]
