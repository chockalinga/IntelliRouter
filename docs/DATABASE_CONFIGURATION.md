# Database Configuration for IntelliRouter

## Why Do We Need Database Configuration ?

IntelliRouter requires database configuration for several critical functions that make it production-ready and scalable. Here's why:

## 🗄️ **Core Database Requirements**

### 1. **Analytics & Metrics Storage**
```sql
-- Track every request for cost optimization
CREATE TABLE request_analytics (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    user_id VARCHAR(255),
    model_selected VARCHAR(100),
    cost_usd DECIMAL(10,6),
    latency_ms INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    task_type VARCHAR(50),
    success BOOLEAN
);
```

**Why needed:**
- Track costs per user/organization
- Identify usage patterns for optimization
- Generate billing reports
- Monitor model performance trends
- Detect anomalies and issues

### 2. **Model Registry & Metadata**
```sql
-- Store dynamic model information
CREATE TABLE models (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255),
    provider VARCHAR(50),
    input_price_per_1k DECIMAL(8,4),
    output_price_per_1k DECIMAL(8,4),
    context_window INTEGER,
    capabilities JSON,
    quality_scores JSON,
    availability_status VARCHAR(20),
    last_updated TIMESTAMP
);
```

**Why needed:**
- Dynamic model pricing updates
- Real-time availability status
- Quality score adjustments based on performance
- A/B testing new models
- Regional model availability

### 3. **User Sessions & Preferences**
```sql
-- Maintain user context and preferences
CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY,
    user_id VARCHAR(255),
    routing_preferences JSON,
    conversation_context JSON,
    budget_limits JSON,
    created_at TIMESTAMP,
    last_activity TIMESTAMP
);
```

**Why needed:**
- Sticky routing (same model for conversation continuity)
- User-specific cost limits and preferences
- Personalized routing policies
- Conversation context for better routing decisions

### 4. **Request Logging & Audit Trail**
```sql
-- Complete request logging for compliance
CREATE TABLE request_logs (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    user_id VARCHAR(255),
    request_data JSON,
    response_data JSON,
    routing_decision JSON,
    error_details TEXT,
    ip_address INET,
    user_agent TEXT
);
```

**Why needed:**
- Compliance and audit requirements
- Debugging and troubleshooting
- Performance analysis
- Security monitoring
- Rate limiting enforcement

### 5. **Configuration Management**
```sql
-- Store routing policies and system config
CREATE TABLE routing_configs (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    user_id VARCHAR(255),
    policy_json JSON,
    is_active BOOLEAN,
    created_at TIMESTAMP
);
```

**Why needed:**
- Multiple routing strategies per user
- A/B testing different policies
- Backup and versioning of configurations
- Environment-specific settings

## 📊 **Real-World Example: Why This Matters**

### Without Database (Limited):
```python
# Current basic approach - loses data on restart
class SimpleRouter:
    def __init__(self):
        self.stats = {}  # Lost when server restarts!
        self.models = hardcoded_models  # Can't update without code change
    
    def route_request(self, request):
        # No historical data for optimization
        # No user preferences
        # No cost tracking
        pass
```

### With Database (Production-Ready):
```python
class ProductionRouter:
    def __init__(self, db_connection):
        self.db = db_connection
        self.analytics = AnalyticsService(db)
        self.model_registry = DatabaseModelRegistry(db)
    
    async def route_request(self, request):
        # 1. Check user preferences from database
        user_prefs = await self.db.get_user_preferences(request.user_id)
        
        # 2. Get real-time model data
        available_models = await self.model_registry.get_available_models()
        
        # 3. Consider historical performance
        model_performance = await self.analytics.get_model_performance()
        
        # 4. Make intelligent routing decision
        selected_model = self.select_optimal_model(
            request, user_prefs, available_models, model_performance
        )
        
        # 5. Log everything for future optimization
        await self.analytics.log_request(request, selected_model)
        
        return selected_model
```

## 🚀 **Production Benefits**

### 1. **Cost Optimization**
```python
# Find which models are most cost-effective for each task type
query = """
SELECT 
    task_type,
    model_selected,
    AVG(cost_usd) as avg_cost,
    AVG(tokens_output/tokens_input) as efficiency_ratio
FROM request_analytics 
WHERE success = true
GROUP BY task_type, model_selected
ORDER BY task_type, avg_cost;
"""

# Result: Automatically optimize routing policies based on real data
```

### 2. **Scalability & Performance**
```python
# Cache frequently used model metadata
@cached(ttl=300)  # 5-minute cache
async def get_model_metadata(model_id):
    return await db.fetch_model_metadata(model_id)

# Rate limiting per user
async def check_rate_limit(user_id):
    recent_requests = await db.count_recent_requests(user_id, minutes=60)
    return recent_requests < USER_HOURLY_LIMIT
```

### 3. **Business Intelligence**
```python
# Generate insights for business decisions
async def generate_usage_report(organization_id, date_range):
    return await db.execute("""
        SELECT 
            DATE(timestamp) as date,
            SUM(cost_usd) as daily_cost,
            COUNT(*) as request_count,
            AVG(latency_ms) as avg_latency
        FROM request_analytics 
        WHERE organization_id = $1 
        AND timestamp BETWEEN $2 AND $3
        GROUP BY DATE(timestamp)
        ORDER BY date;
    """, organization_id, date_range.start, date_range.end)
```

## 🔧 **Database Choice Considerations**

### **PostgreSQL (Recommended)**
```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: intellirouter
      POSTGRES_USER: intellirouter
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
```

**Why PostgreSQL:**
- JSON support for flexible schemas
- ACID compliance for critical data
- Great performance for analytics queries
- Rich ecosystem and tooling

### **Alternative: MongoDB**
```python
# For more flexible document storage
{
    "_id": "request_123",
    "timestamp": "2025-01-08T10:00:00Z",
    "user_id": "user_456",
    "routing_decision": {
        "selected_model": "gpt-4o-mini",
        "reason": "cost optimization",
        "alternatives_considered": ["gpt-4o", "claude-3-haiku"],
        "decision_factors": {
            "cost_weight": 0.6,
            "quality_weight": 0.4
        }
    },
    "performance_metrics": {
        "cost_usd": 0.000032,
        "latency_ms": 850,
        "tokens": {"input": 20, "output": 45}
    }
}
```

## 🛠️ **Configuration Setup**

### Environment Variables:
```bash
# .env
DATABASE_URL=postgresql://intellirouter:password@localhost:5432/intellirouter
REDIS_URL=redis://localhost:6379/0  # For caching
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
ANALYTICS_RETENTION_DAYS=90
```

### Application Configuration:
```python
# src/intellirouter/config.py
class DatabaseConfig:
    url: str = os.getenv("DATABASE_URL")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "20"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "30"))
    echo_queries: bool = os.getenv("DB_ECHO", "false").lower() == "true"
    
    # Analytics settings
    retention_days: int = int(os.getenv("ANALYTICS_RETENTION_DAYS", "90"))
    enable_request_logging: bool = True
    enable_performance_tracking: bool = True
```

## 📈 **Migration Strategy**

### Phase 1: Basic Analytics
```sql
-- Start with essential tables
CREATE TABLE request_analytics (...);
CREATE TABLE user_sessions (...);
```

### Phase 2: Advanced Features
```sql
-- Add advanced analytics
CREATE TABLE model_performance_history (...);
CREATE TABLE routing_experiments (...);
```

### Phase 3: Enterprise Features
```sql
-- Add enterprise capabilities
CREATE TABLE organizations (...);
CREATE TABLE billing_records (...);
CREATE TABLE compliance_logs (...);
```

## 🔍 **Monitoring & Maintenance**

### Database Health Checks:
```python
async def health_check():
    """Check database connectivity and performance."""
    try:
        # Connection test
        await db.execute("SELECT 1")
        
        # Performance test
        start_time = time.time()
        await db.execute("SELECT COUNT(*) FROM request_analytics")
        query_time = time.time() - start_time
        
        return {
            "database": "healthy",
            "connection": "ok",
            "query_performance_ms": query_time * 1000
        }
    except Exception as e:
        return {"database": "unhealthy", "error": str(e)}
```

### Automated Cleanup:
```python
async def cleanup_old_data():
    """Remove old analytics data based on retention policy."""
    cutoff_date = datetime.now() - timedelta(days=config.retention_days)
    
    deleted_count = await db.execute("""
        DELETE FROM request_analytics 
        WHERE timestamp < $1
    """, cutoff_date)
    
    logger.info(f"Cleaned up {deleted_count} old analytics records")
```

## 🎯 **Summary: Why Database Configuration is Essential**

1. **🔍 Analytics**: Track costs, performance, and usage patterns
2. **👤 User Management**: Handle preferences, sessions, and billing  
3. **⚡ Performance**: Caching, rate limiting, and optimization
4. **🛡️ Compliance**: Audit trails and data governance
5. **📊 Business Intelligence**: Insights for decision making
6. **🔧 Scalability**: Handle enterprise-level traffic and data

**Without a database, IntelliRouter is just a stateless proxy. With a database, it becomes an intelligent, learning, optimizing system that gets better over time.**

The database is what transforms IntelliRouter from a simple routing tool into a comprehensive AI model management platform.
