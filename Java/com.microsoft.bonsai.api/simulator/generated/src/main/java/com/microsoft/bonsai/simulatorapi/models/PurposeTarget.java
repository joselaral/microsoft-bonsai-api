/**
 * Code generated by Microsoft (R) AutoRest Code Generator.
 * Changes may cause incorrect behavior and will be lost if the code is
 * regenerated.
 */

package com.microsoft.bonsai.simulatorapi.models;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * This structure describes the "target" of the simulator;
 * i.e., what trainable construct(s) it exists to service.
 */
public class PurposeTarget {
    /**
     * The workspaceName property.
     */
    @JsonProperty(value = "workspaceName")
    private String workspaceName;

    /**
     * This is the brain _short_ name!.
     */
    @JsonProperty(value = "brainName")
    private String brainName;

    /**
     * The brainVersion property.
     */
    @JsonProperty(value = "brainVersion")
    private Integer brainVersion;

    /**
     * The conceptName property.
     */
    @JsonProperty(value = "conceptName")
    private String conceptName;

    /**
     * Get the workspaceName value.
     *
     * @return the workspaceName value
     */
    public String workspaceName() {
        return this.workspaceName;
    }

    /**
     * Set the workspaceName value.
     *
     * @param workspaceName the workspaceName value to set
     * @return the PurposeTarget object itself.
     */
    public PurposeTarget withWorkspaceName(String workspaceName) {
        this.workspaceName = workspaceName;
        return this;
    }

    /**
     * Get this is the brain _short_ name!.
     *
     * @return the brainName value
     */
    public String brainName() {
        return this.brainName;
    }

    /**
     * Set this is the brain _short_ name!.
     *
     * @param brainName the brainName value to set
     * @return the PurposeTarget object itself.
     */
    public PurposeTarget withBrainName(String brainName) {
        this.brainName = brainName;
        return this;
    }

    /**
     * Get the brainVersion value.
     *
     * @return the brainVersion value
     */
    public Integer brainVersion() {
        return this.brainVersion;
    }

    /**
     * Set the brainVersion value.
     *
     * @param brainVersion the brainVersion value to set
     * @return the PurposeTarget object itself.
     */
    public PurposeTarget withBrainVersion(Integer brainVersion) {
        this.brainVersion = brainVersion;
        return this;
    }

    /**
     * Get the conceptName value.
     *
     * @return the conceptName value
     */
    public String conceptName() {
        return this.conceptName;
    }

    /**
     * Set the conceptName value.
     *
     * @param conceptName the conceptName value to set
     * @return the PurposeTarget object itself.
     */
    public PurposeTarget withConceptName(String conceptName) {
        this.conceptName = conceptName;
        return this;
    }

}