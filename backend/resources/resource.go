package resources

import "github.com/gin-gonic/gin"

type Resource struct {
	appliedMiddleware []gin.HandlerFunc
}
