
SAMPLE =[
    {
        "id": "TICKET-245",
        "subject": "Connecting Snowflake to Atlan - required permissions?",
        "body": "Hi team, we're trying to set up our primary Snowflake production database as a new source in Atlan, but the connection keeps failing. We've tried using our standard service account, but it's not working. Our entire BI team is blocked on this integration for a major upcoming project, so it's quite urgent. Could you please provide a definitive list of the exact permissions and credentials needed on the Snowflake side to get this working? Thanks."
    },
    {
        "id": "TICKET-246",
        "subject": "Which connectors automatically capture lineage?",
        "body": "Hello, I'm new to Atlan and trying to understand the lineage capabilities. The documentation mentions automatic lineage, but it's not clear which of our connectors (we use Fivetran, dbt, and Tableau) support this out-of-the-box. We need to present a clear picture of our data flow to leadership next week. Can you explain how lineage capture differs for these tools?"
    },
    {
        "id": "TICKET-247",
        "subject": "Deployment of Atlan agent for private data lake",
        "body": "Our primary data lake is hosted on-premise within a secure VPC and is not exposed to the internet. We understand we need to use the Atlan agent for this, but the setup instructions are a bit confusing for our security team. This is a critical source for us, and we can't proceed with our rollout until we get this connected. Can you provide a detailed deployment guide or connect us with a technical expert?"
    },
    {
        "id": "TICKET-248",
        "subject": "How to surface sample rows and schema changes?",
        "body": "Hi, we've successfully connected our Redshift cluster, and the assets are showing up. However, my data analysts are asking how they can see sample data or recent schema changes directly within Atlan without having to go back to Redshift. Is this feature available? I feel like I'm missing something obvious."
    },
    {
        "id": "TICKET-249",
        "subject": "Exporting lineage view for a specific table",
        "body": "For our quarterly audit, I need to provide a complete upstream and downstream lineage diagram for our core `fact_orders` table. I can see the lineage perfectly in the UI, but I can't find an option to export this view as an image or PDF. This is a hard requirement from our compliance team and the deadline is approaching fast. Please help!"
    },
    {
        "id": "TICKET-250",
        "subject": "Importing lineage from Airflow jobs",
        "body": "We run hundreds of ETL jobs in Airflow, and we need to see that lineage reflected in Atlan. I've read that Atlan can integrate with Airflow, but how do we configure it to correctly map our DAGs to the specific datasets they are transforming? The current documentation is a bit high-level."
    },
    {
        "id": "TICKET-251",
        "subject": "Using the Visual Query Builder",
        "body": "I'm a business analyst and not very comfortable with writing complex SQL. I was excited to see the Visual Query Builder in Atlan, but I'm having trouble figuring out how to join multiple tables and save my query for later use. Is there a tutorial or a quick guide you can point me to?"
    },
    {
        "id": "TICKET-252",
        "subject": "Programmatic extraction of lineage",
        "body": "Our internal data science team wants to build a custom application that analyzes metadata propagation delays. To do this, we need to programmatically extract lineage data from Atlan via an API. Does the API expose lineage information, and if so, could you provide an example of the endpoint and the structure of the response?"
    },
    {
        "id": "TICKET-253",
        "subject": "Upstream lineage to Snowflake view not working",
        "body": "This is infuriating. We have a critical Snowflake view, `finance.daily_revenue`, that is built from three upstream tables. Atlan is correctly showing the downstream dependencies, but the upstream lineage is completely missing. This makes the view untrustworthy for our analysts. We've re-run the crawler multiple times. What could be causing this? This is a huge problem for us."
    },
    {
        "id": "TICKET-254",
        "subject": "How to create a business glossary and link terms in bulk?",
        "body": "We are migrating our existing business glossary from a spreadsheet into Atlan. We have over 500 terms. Manually creating each one and linking them to thousands of assets seems impossible. Is there a bulk import feature using CSV or an API to create terms and link them to assets? This is blocking our entire governance initiative."
    },
    {
        "id": "TICKET-255",
        "subject": "Creating a custom role for data stewards",
        "body": "I'm trying to set up a custom role for our data stewards. They need permission to edit descriptions and link glossary terms, but they should NOT have permission to run queries or change connection settings. I'm looking at the default roles, but none of them fit perfectly. How can I create a new role with this specific set of permissions?"
    },
    {
        "id": "TICKET-256",
        "subject": "Mapping Active Directory groups to Atlan teams",
        "body": "Our company policy requires us to manage all user access through Active Directory groups. We need to map our existing AD groups (e.g., 'data-analyst-finance', 'data-engineer-core') to teams within Atlan to automatically grant the correct permissions. I can't find the settings for this. How is this configured?"
    },
    {
        "id": "TICKET-257",
        "subject": "RBAC for assets vs. glossaries",
        "body": "I need clarification on how Atlan's role-based access control works. If a user is denied access to a specific Snowflake schema, can they still see the glossary terms that are linked to the tables in that schema? I need to ensure our PII governance is airtight."
    },
    {
        "id": "TICKET-258",
        "subject": "Process for onboarding asset owners",
        "body": "We've started identifying owners for our key data assets. What is the recommended workflow in Atlan to assign these owners and automatically notify them? We want to make sure they are aware of their responsibilities without us having to send manual emails for every assignment."
    },
    {
        "id": "TICKET-259",
        "subject": "How does Atlan surface sensitive fields like PII?",
        "body": "Our security team is evaluating Atlan and their main question is around PII and sensitive data. How does Atlan automatically identify fields containing PII? What are our options to apply tags or masks to these fields once they are identified to prevent unauthorized access?"
    },
    {
        "id": "TICKET-260",
        "subject": "Authentication methods for APIs and SDKs",
        "body": "We are planning to build several automations using the Atlan API and Python SDK. What authentication methods are supported? Is it just API keys, or can we use something like OAuth? We have a strict policy that requires key rotation every 90 days, so we need to understand how to manage this programmatically."
    },
    {
        "id": "TICKET-261",
        "subject": "Enabling and testing SAML SSO",
        "body": "We are ready to enable SAML SSO with our Okta instance. However, we are very concerned about disrupting our active users if the configuration is wrong. Is there a way to test the SSO configuration for a specific user or group before we enable it for the entire workspace?"
    },
    {
        "id": "TICKET-262",
        "subject": "SSO login not assigning user to correct group",
        "body": "I've just had a new user, 'test.user@company.com', log in via our newly configured SSO. They were authenticated successfully, but they were not added to the 'Data Analysts' group as expected based on our SAML assertions. This is preventing them from accessing any assets. What could be the reason for this mis-assignment?"
    },
    {
        "id": "TICKET-263",
        "subject": "Integration with existing DLP or secrets manager",
        "body": "Does Atlan have the capability to integrate with third-party tools like a DLP (Data Loss Prevention) solution or a secrets manager like HashiCorp Vault? We need to ensure that connection credentials and sensitive metadata classifications are handled by our central security systems."
    },
    {
        "id": "TICKET-264",
        "subject": "Accessing audit logs for compliance reviews",
        "body": "Our compliance team needs to perform a quarterly review of all activities within Atlan. They need to know who accessed what data, who made permission changes, etc. Where can we find these audit logs, and is there a way to export them or pull them via an API for our records?"
    },
    {
        "id": "TICKET-265",
        "subject": "How to programmatically create an asset using the REST API?",
        "body": "I'm trying to create a new custom asset (a 'Report') using the REST API, but my requests keep failing with a 400 error. The API documentation is a bit sparse on the required payload structure for creating new entities. Could you provide a basic cURL or Python `requests` example of what a successful request body should look like?"
    },
    {
        "id": "TICKET-266",
        "subject": "SDK availability and Python example",
        "body": "I'm a data engineer and prefer using SDKs over raw API calls. Which languages do you provide SDKs for? I'm particularly interested in Python. Where can I find the installation instructions (e.g., PyPI package name) and a short code snippet for a common task, like creating a new glossary term?"
    },
    {
        "id": "TICKET-267",
        "subject": "How do webhooks work in Atlan?",
        "body": "I'm exploring using webhooks to send real-time notifications from Atlan to our internal Slack channel. I need to understand what types of events (e.g., asset updated, term created) can trigger a webhook. Also, how do we validate that the incoming payloads are genuinely from Atlan? Do you support payload signing?"
    },
    {
        "id": "TICKET-268",
        "subject": "Triggering an AWS Lambda from Atlan events",
        "body": "We have a workflow where we want to trigger a custom AWS Lambda function whenever a specific Atlan tag (e.g., 'PII-Confirmed') is added to an asset. What is the recommended and most secure way to set this up? Should we use webhooks pointing to an API Gateway, or is there a more direct integration?"
    },
    {
        "id": "TICKET-269",
        "subject": "When to use Atlan automations vs. external services?",
        "body": "I see that Atlan has a built-in 'Automations' feature. I'm trying to decide if I should use this to manage a workflow or if I should use an external service like Zapier or our own Airflow instance. Could you provide some guidance or examples on what types of workflows are best suited for the native automations versus an external tool?"
    },
    {
        "id": "TICKET-270",
        "subject": "Connector failed to crawl - where to check logs?",
        "body": "URGENT: Our nightly Snowflake crawler failed last night and no new metadata was ingested. This is a critical failure as our morning reports are now missing lineage information. Where can I find the detailed error logs for the crawler run to understand what went wrong? I need to fix this ASAP."
    },
    {
        "id": "TICKET-271",
        "subject": "Asset extracted but not published to Atlan",
        "body": "This is very strange. I'm looking at the crawler logs, and I can see that the asset 'schema.my_table' was successfully extracted from the source. However, when I search for this table in the Atlan UI, it doesn't appear. It seems like it's getting stuck somewhere between extraction and publishing. Can you please investigate the root cause?"
    },
    {
        "id": "TICKET-272",
        "subject": "How to measure adoption and generate reports?",
        "body": "My manager is asking for metrics on our Atlan usage to justify the investment. I need to generate a report showing things like the number of active users, most frequently queried tables, and the number of assets with assigned owners. Does Atlan have a reporting or dashboarding feature for this?"
    },
    {
        "id": "TICKET-273",
        "subject": "Best practices for catalog hygiene",
        "body": "We've been using Atlan for six months, and our catalog is already starting to get a bit messy with duplicate assets and stale metadata from old tests. As we roll this out to more teams, what are some common best practices or features within Atlan that can help us maintain good catalog hygiene and prevent this problem from getting worse?"
    },
    {
        "id": "TICKET-274",
        "subject": "How to scale Atlan across multiple business units?",
        "body": "We are planning a global rollout of Atlan to multiple business units, each with its own data sources and governance teams. We're looking for advice on the best way to structure our Atlan instance. Should we use separate workspaces, or can we achieve isolation using teams and permissions within a single workspace while maintaining a consistent governance model?"
    }
]