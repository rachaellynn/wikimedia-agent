Core LangGraph Architecture

  1. State Management (src/agent.py:40-55)
  - Uses AgentState TypedDict to define data flow through workflow
  - State persists through all workflow steps: input → processing → output
  - Contains subject, images, errors, and final results

  2. Graph Construction (src/agent.py:64-98)
  workflow = StateGraph(AgentState)
  workflow.add_node("search_wikimedia", self._search_wikimedia)
  workflow.add_node("curate_images", self._curate_images)
  workflow.add_node("save_to_mongodb", self._save_to_mongodb)
  workflow.add_node("handle_error", self._handle_error)

  3. Workflow Orchestration
  - Linear flow: search_wikimedia → curate_images → save_to_mongodb
  - Error handling: Conditional edges route to error handler on failures
  - State updates: Each node receives current state, processes it, returns updated state

  Workflow Steps

  Step 1: Wikimedia Search (src/agent.py:100-160)
  - Queries Wikimedia Commons API via src/tools/wikimedia.py
  - Handles both primary photos ("Einstein") and context photos ("Einstein context historical")
  - Fallback logic if enhanced query fails
  - Updates state with raw_images

  Step 2: AI-Powered Curation (workflows/image_curator.py)
  - Filtering: Removes videos/invalid formats (workflows/image_curator.py:41-51)
  - Deduplication: Smart title/URL comparison (workflows/image_curator.py:279-349)
  - AI Analysis: OpenAI Vision API scores each image (workflows/image_curator.py:130-253)
    - Portrait quality (0-10)
    - Historical significance (0-10)
    - Technical quality (0-10)
  - Selection: Best portrait → featured, rest → carousel

  Step 3: Database Storage (src/agent.py:225-300)
  - Saves to MongoDB with rich metadata
  - Each image becomes separate document in "images" collection
  - Includes AI scores, dimensions, attribution, timestamps

  LangChain Integration

  Error Handling & Conditional Logic
  workflow.add_conditional_edges(
      "search_wikimedia",
      self._check_search_success,
      {"success": "curate_images", "error": "handle_error"}
  )

  State Persistence
  - State flows through entire pipeline automatically
  - Each function receives current state, returns updated state
  - LangGraph handles state transitions and error recovery

  Key Design Patterns

  1. Separation of Concerns
  - src/agent.py: Workflow orchestration
  - workflows/image_curator.py: AI analysis logic
  - src/tools/wikimedia.py: External API integration
  - src/utils/: Database and OpenAI helpers

  2. Defensive Programming
  - Extensive error handling at each step
  - Fallback queries if initial search fails
  - Graceful degradation when AI analysis fails

  3. Async/Await Pattern
  - All workflow functions are async
  - Supports concurrent operations and better performance

  This architecture provides a robust, maintainable system for AI-powered image curation with
  clear separation of concerns and comprehensive error handling.